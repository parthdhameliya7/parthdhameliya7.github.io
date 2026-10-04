---
layout: page
title: About
permalink: /about/
subtitle: Data Scientist @ XPO. Papers, reproduced.
published: false
---

<div class="about-profile">
  <img
    class="about-profile__avatar"
    src="{{ site.avatar | relative_url }}"
    alt="{{ site.author.name }}"
    width="140"
    height="168"
  >
  <div>
    <p class="about-profile__name">{{ site.author.name }}</p>
    {% if site.author.location != "" and site.author.location != "City, Country" %}
      <p class="about-profile__meta">{{ site.author.location }}</p>
    {% endif %}
  </div>
</div>

Hi — I'm **{{ site.author.name }}**, a Data Scientist at XPO.

I reproduce the results of state-of-the-art research papers, then write about what it takes to get there: setups, pitfalls, and the details that don’t always make it into the PDF.

I write about:

- Reproducing paper experiments end to end
- What breaks in practice — and how to fix it
- Clear explanations of new ideas in AI

If you want to get in touch, use the links in the footer (set them in `_config.yml`).
