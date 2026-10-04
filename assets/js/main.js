(() => {
  const root = document.documentElement;
  const toggle = document.querySelector("[data-theme-toggle]");

  const applyTheme = (theme) => {
    root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("theme", theme);
    } catch (e) {
      /* ignore */
    }
    if (toggle) {
      const next = theme === "dark" ? "light" : "dark";
      toggle.setAttribute("aria-label", `Switch to ${next} theme`);
    }
  };

  const current = root.getAttribute("data-theme") || "light";
  applyTheme(current);

  if (toggle) {
    toggle.addEventListener("click", () => {
      applyTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
    });
  }

  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const nodes = document.querySelectorAll("[data-animate]");

  if (reduce || !("IntersectionObserver" in window)) {
    nodes.forEach((el) => el.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0, rootMargin: "0px 0px -40px 0px" }
  );

  nodes.forEach((el) => observer.observe(el));
})();
