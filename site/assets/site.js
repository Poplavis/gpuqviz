/* gpuqviz 站点共享脚本：导航栏、画廊渲染、标签云、Modal */

(function () {
  "use strict";

  // ─── 导航栏汉堡菜单 ───
  function initNavbar() {
    var toggle = document.querySelector(".nav-toggle");
    var links = document.querySelector(".nav-links");
    if (!toggle || !links) return;
    toggle.addEventListener("click", function () {
      links.classList.toggle("open");
    });
  }

  // ─── 画廊渲染 ───
  function renderGallery() {
    var grid = document.getElementById("galleryGrid");
    if (!grid) return;
    if (!window.GPUQVIZ_GALLERY) {
      grid.innerHTML = '<p style="color:var(--dim);text-align:center;">'
        + '请先运行 <code>python scripts/build_site.py</code> 生成画廊数据。</p>';
      return;
    }

    var html = "";
    window.GPUQVIZ_GALLERY.forEach(function (algo) {
      var thumbHtml = algo.thumb_url
        ? '<img src="' + algo.thumb_url + '" alt="' + algo.name + '" loading="lazy">'
        : '<div class="gallery-thumb">' + algo.name.charAt(0).toUpperCase() + "</div>";
      html +=
        '<div class="gallery-card" data-category="' + algo.category + '" data-name="' + algo.name + '">' +
        '<div class="gallery-thumb">' + thumbHtml + "</div>" +
        '<div class="gallery-body">' +
        "<h3>" + algo.name + "</h3>" +
        '<p class="desc">' + algo.description + "</p>" +
        '<div class="gallery-meta">' +
        '<span class="cat">' + algo.category + "</span>" +
        '<span class="qubits">' + algo.n_qubits + " qubit</span>" +
        "</div>" +
        "</div>" +
        "</div>";
    });
    grid.innerHTML = html;

    // 点击卡片 → 打开 Modal
    grid.querySelectorAll(".gallery-card").forEach(function (card) {
      card.addEventListener("click", function () {
        var name = card.getAttribute("data-name");
        openDemoModal("demos/" + name + ".html");
      });
    });

    // URL hash 定位
    if (location.hash) {
      var target = location.hash.substring(1);
      var card = grid.querySelector('[data-name="' + target + '"]');
      if (card) {
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.classList.add("highlight");
      }
    }
  }

  // ─── 画廊分类筛选 ───
  function initGalleryFilters() {
    var buttons = document.querySelectorAll(".filter-btn");
    var cards = document.querySelectorAll(".gallery-card");
    if (!buttons.length || !cards.length) return;

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var cat = btn.getAttribute("data-filter");
        buttons.forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");

        cards.forEach(function (card) {
          if (cat === "all" || card.getAttribute("data-category") === cat) {
            card.style.display = "";
          } else {
            card.style.display = "none";
          }
        });
      });
    });
  }

  // ─── 演示 Modal ───
  function initDemoModal() {
    var modal = document.getElementById("demoModal");
    if (!modal) return;

    var closeBtn = modal.querySelector(".close-btn");
    var iframe = modal.querySelector("iframe");

    closeBtn.addEventListener("click", closeDemoModal);
    modal.addEventListener("click", function (e) {
      if (e.target === modal) closeDemoModal();
    });

    window.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeDemoModal();
    });
  }

  window.openDemoModal = function (url) {
    var modal = document.getElementById("demoModal");
    if (!modal) {
      window.open(url, "_blank");
      return;
    }
    var iframe = modal.querySelector("iframe");
    iframe.src = url;
    modal.classList.add("open");
    document.body.style.overflow = "hidden";
  };

  function closeDemoModal() {
    var modal = document.getElementById("demoModal");
    if (!modal) return;
    var iframe = modal.querySelector("iframe");
    iframe.src = "about:blank";
    modal.classList.remove("open");
    document.body.style.overflow = "";
  }
  window.closeDemoModal = closeDemoModal;

  // ─── 首页算法标签云 ───
  function renderAlgoTags() {
    var container = document.getElementById("algoTags");
    if (!container) return;
    if (!window.GPUQVIZ_GALLERY) return;

    var html = "";
    window.GPUQVIZ_GALLERY.forEach(function (algo) {
      html +=
        '<a class="algo-tag" href="gallery.html#' + algo.name + '">' +
        algo.name + "</a>";
    });
    container.innerHTML = html;
  }

  // ─── 代码块复制按钮 ───
  function initCopyButtons() {
    document.querySelectorAll("pre").forEach(function (pre) {
      var btn = document.createElement("button");
      btn.textContent = "复制";
      btn.className = "copy-btn";
      btn.style.cssText =
        "position:absolute;top:8px;right:8px;padding:4px 12px;" +
        "border:1px solid var(--border);border-radius:6px;background:var(--panel);" +
        "color:var(--dim);font-size:0.8rem;cursor:pointer;opacity:0;transition:opacity .15s;";
      pre.style.position = "relative";
      pre.appendChild(btn);

      pre.addEventListener("mouseenter", function () { btn.style.opacity = "1"; });
      pre.addEventListener("mouseleave", function () { btn.style.opacity = "0"; });
      btn.addEventListener("click", function () {
        var code = pre.querySelector("code");
        if (code && navigator.clipboard) {
          navigator.clipboard.writeText(code.textContent).then(function () {
            btn.textContent = "✓ 已复制";
            setTimeout(function () { btn.textContent = "复制"; }, 1500);
          });
        }
      });
    });
  }

  // ─── 文档侧边栏滚动高亮 ───
  function initDocsSidebar() {
    var links = document.querySelectorAll(".docs-sidebar a");
    var sections = [];
    links.forEach(function (link) {
      var href = link.getAttribute("href");
      if (href && href.startsWith("#")) {
        var el = document.getElementById(href.substring(1));
        if (el) sections.push({ link: link, el: el });
      }
    });
    if (!sections.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            links.forEach(function (l) { l.classList.remove("active"); });
            var match = sections.find(function (s) { return s.el === entry.target; });
            if (match) match.link.classList.add("active");
          }
        });
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );
    sections.forEach(function (s) { observer.observe(s.el); });
  }

  // ─── 初始化 ───
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  function init() {
    initNavbar();
    renderGallery();
    initGalleryFilters();
    initDemoModal();
    renderAlgoTags();
    initCopyButtons();
    initDocsSidebar();
  }
})();
