"""Subtitle layout preserving word timings when present."""
import copy
import textwrap
import re
import unicodedata


def _normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _wrap(text, width):
    return textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [text]


def layout_segments(segments, config):
    width = int(config.get("max_line_length", 42))
    max_lines = int(config.get("max_lines", 2))
    max_duration = float(config.get("max_cue_seconds", 7))
    result = []
    estimated = False
    for original in segments:
        segment = copy.deepcopy(original)
        if not segment.get("text", "").strip():
            continue
        words = [w for w in segment.get("words", []) if w.get("start") is not None and w.get("end") is not None and w.get("word", "").strip()]
        text = segment["text"].strip()
        # Alignment may omit numbers, names or whole words. Never rebuild the
        # transcript from incomplete timestamps, which would silently lose text.
        complete = words and _normalized("".join(w["word"] for w in words)) == _normalized(text)
        spans, cursor = [], 0
        if complete:
            for word in words:
                start = text.find(word["word"].strip(), cursor)
                if start < 0:
                    complete = False
                    break
                cursor = start + len(word["word"].strip())
                spans.append((start, cursor))
        if complete:
            groups, group = [], []
            for index, word in enumerate(words):
                candidate = text[spans[group[0]][0] if group else spans[index][0]:spans[index][1]]
                if group and (len(_wrap(candidate, width)) > max_lines or word["end"] - words[group[0]]["start"] > max_duration):
                    groups.append(group)
                    group = []
                group.append(index)
            if group:
                groups.append(group)
            pieces = [{**segment, "start": words[group[0]]["start"], "end": words[group[-1]]["end"], "text": text[spans[group[0]][0]:spans[group[-1]][1]], "words": [words[i] for i in group]} for group in groups]
        else:
            if words:
                estimated = True
            lines = _wrap(text, width)
            blocks = ["\n".join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)]
            duration = max(0.01, segment["end"] - segment["start"])
            length = max(1, sum(len(b) for b in blocks))
            elapsed = segment["start"]
            pieces = []
            for block in blocks:
                end = elapsed + duration * len(block) / length
                pieces.append({**segment, "start": elapsed, "end": end, "text": block, "words": []})
                elapsed = end
            if len(blocks) > 1:
                estimated = True
        for piece in pieces:
            lines = _wrap(piece["text"].replace("\n", " "), width)
            piece["text"] = "\n".join(lines)
            offset = float(config.get("time_offset", 0))
            piece["start"] = max(0.0, piece["start"] + offset)
            piece["end"] = max(0.0, piece["end"] + offset)
            if piece["end"] > piece["start"]:
                result.append(piece)
    result.sort(key=lambda s: (s["start"], s["end"]))
    minimum = float(config.get("min_cue_seconds", 0.5))
    for index, item in enumerate(result):
        next_start = result[index + 1]["start"] if index + 1 < len(result) else float("inf")
        if item["end"] - item["start"] < minimum and next_start > item["end"]:
            item["end"] = min(item["start"] + minimum, next_start)
        item["id"] = index + 1
    return result, estimated
