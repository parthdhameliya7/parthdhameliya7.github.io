---
layout: page
title: Archive
permalink: /archive/
subtitle: Every post, newest first.
---

{% assign posts_by_year = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}

{% if site.posts.size == 0 %}
<p class="empty-state">No posts yet.</p>
{% else %}
  {% for year in posts_by_year %}
    <section class="archive-year">
      <h2 class="archive-year__title">{{ year.name }}</h2>
      <ul class="post-list post-list--compact">
        {% for post in year.items %}
          <li class="post-list__item">
            <a class="post-list__link" href="{{ post.url | relative_url }}">
              <time class="post-list__date" datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d" }}</time>
              <span class="post-list__title">{{ post.title }}</span>
            </a>
          </li>
        {% endfor %}
      </ul>
    </section>
  {% endfor %}
{% endif %}
