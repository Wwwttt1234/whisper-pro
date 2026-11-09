"""Consistent explanations on setting labels, including keyboard focus."""
import hashlib
import html
import streamlit as st

CSS = """<style>
.setting-label{position:relative;font-weight:600;padding:.25rem 0;cursor:help;outline-offset:3px}
.setting-label .hint-icon{opacity:.7;font-size:.85em}
.setting-label .hint-bubble{visibility:hidden;opacity:0;position:absolute;inset-inline-start:0;top:100%;z-index:9999;
width:min(360px,78vw);max-width:100%;padding:.8rem;border:1px solid #6c839b;border-radius:10px;
background:#142436;color:#f2f6fb;box-shadow:0 6px 20px #0005;font-weight:400;font-size:.9rem;line-height:1.65;white-space:pre-wrap;pointer-events:none}
.setting-label:hover .hint-bubble,.setting-label:focus .hint-bubble{visibility:visible;opacity:1}
</style>"""


def tooltip_label(label, help_text, container=None):
    target = container or st
    identity = "hint-" + hashlib.sha256((label + help_text).encode()).hexdigest()[:16]
    target.html(f'<div class="setting-label" tabindex="0" aria-describedby="{identity}">{html.escape(label)} <span class="hint-icon">ⓘ</span><span class="hint-bubble" id="{identity}" role="tooltip">{html.escape(help_text)}</span></div>')


class ExplainedControls:
    """Keep normal Streamlit widgets and keys; add hover labels to their help."""
    def __getattr__(self, name):
        def widget(label, *args, _container=None, **kwargs):
            target = _container or st
            if kwargs.get("help") and kwargs.get("label_visibility") != "collapsed":
                tooltip_label(label, kwargs["help"], target)
                kwargs["label_visibility"] = "collapsed"
            kwargs.pop("help", None)
            return getattr(target, name)(label, *args, **kwargs)
        return widget


controls = ExplainedControls()
