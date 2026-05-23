// Service Worker v3 — 智能缓存 + 离线优先
var CACHE_STATIC = 'ai-skin-static-v3';
var CACHE_DYNAMIC = 'ai-skin-dynamic-v3';

// 静态资源（版本号库文件用 immutable 策略，永远不重新请求）
var IMMUTABLE_URLS = [
  '/lib/tf.min.js',
  '/lib/blazeface.min.js'
];

var PRECACHE_URLS = [
  '/',
  '/index.html',
  '/css/style.css',
  '/js/app.js'
].concat(IMMUTABLE_URLS);

// ===== Install: 预缓存核心资源 =====
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_STATIC).then(function(cache) {
      console.log('[SW v3] Pre-caching ' + PRECACHE_URLS.length + ' assets');
      return Promise.allSettled(PRECACHE_URLS.map(function(url) {
        return cache.add(url).catch(function(err) {
          console.warn('[SW v3] Failed to pre-cache:', url, err.message);
        });
      }));
    })
  );
  self.skipWaiting();
});

// ===== Activate: 清理旧版本缓存 =====
self.addEventListener('activate', function(event) {
  event.waitUntil(
    caches.keys().then(function(keys) {
      return Promise.all(
        keys.filter(function(key) {
          return key !== CACHE_STATIC && key !== CACHE_DYNAMIC;
        }).map(function(key) {
          console.log('[SW v3] Deleting old cache:', key);
          return caches.delete(key);
        })
      );
    })
  );
  self.clients.claim();
});

// ===== Fetch: 分层缓存策略 =====
self.addEventListener('fetch', function(event) {
  var req = event.request;
  var url = new URL(req.url);

  // 1. 版本库文件：Cache-only（immutable，永不更新）
  if (url.pathname.indexOf('/lib/') !== -1) {
    event.respondWith(
      caches.match(req).then(function(cached) {
        return cached || fetch(req).then(function(response) {
          var clone = response.clone();
          caches.open(CACHE_STATIC).then(function(c) { c.put(req, clone); });
          return response;
        });
      })
    );
    return;
  }

  // 2. CSS/JS：Cache-first（版本号变化时 SW 更新即可）
  if (url.pathname.indexOf('/css/') !== -1 || url.pathname.indexOf('/js/') !== -1) {
    event.respondWith(
      caches.match(req).then(function(cached) {
        return cached || fetch(req).then(function(response) {
          caches.open(CACHE_STATIC).then(function(c) { c.put(req, response.clone()); });
          return response;
        });
      })
    );
    return;
  }

  // 3. HTML：Network-first（始终获取最新，失败时回退缓存）
  if (req.destination === 'document') {
    event.respondWith(
      fetch(req).catch(function() {
        return caches.match(req);
      })
    );
    return;
  }

  // 4. 其他资源：Stale-while-revalidate
  event.respondWith(
    caches.match(req).then(function(cached) {
      var fetched = fetch(req).then(function(response) {
        caches.open(CACHE_DYNAMIC).then(function(c) { c.put(req, response.clone()); });
        return response;
      }).catch(function() { return cached; });
      return cached || fetched;
    })
  );
});

// Message: 支持客户端触发缓存更新
self.addEventListener('message', function(event) {
  if (event.data && event.data.action === 'skipWaiting') {
    self.skipWaiting();
  }
});
