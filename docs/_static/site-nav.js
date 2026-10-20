// Shared tab line: theme toggle (kept in sync with the benchmark explorer and the docs theme) and menus.
(function () {
  var root = document.documentElement;
  var nav = document.querySelector(".site-nav");
  var button = document.getElementById("theme");
  function label() {
    if (!button) return;
    var next = root.dataset.theme === "dark" ? "Light" : "Dark";
    button.textContent = next + " mode";
    button.setAttribute("aria-label", "Switch to " + next.toLowerCase() + " mode");
  }
  label();
  new MutationObserver(label).observe(root, { attributes: true, attributeFilter: ["data-theme", "data-mode"] });
  if (button)
    button.addEventListener("click", function () {
      var theme = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = theme;
      root.dataset.mode = theme; // the docs theme reads data-mode as well
      try {
        localStorage.setItem("rfd4-proteina-theme", theme);
        localStorage.setItem("mode", theme);
      } catch (e) {}
      document.dispatchEvent(new Event("themechange"));
      window.dispatchEvent(new Event("site-theme-change"));
    });
  var toggle = document.querySelector(".site-nav-toggle");
  if (toggle && nav)
    toggle.addEventListener("click", function () {
      nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", nav.classList.contains("open"));
    });
  document.querySelectorAll(".tab-caret").forEach(function (caret) {
    caret.addEventListener("click", function (event) {
      var group = caret.closest(".tab-group");
      var open = group.classList.toggle("open");
      caret.setAttribute("aria-expanded", open);
      event.stopPropagation();
    });
  });
  document.addEventListener("click", function () {
    document.querySelectorAll(".tab-group.open").forEach(function (g) { g.classList.remove("open"); });
  });
  document.querySelectorAll(".tab-menu a").forEach(function (a) {
    a.addEventListener("click", function () { if (nav) nav.classList.remove("open"); });
  });
})();
