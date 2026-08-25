"""Build the GitHub Pages site for weekly rankings.

Copies all ranking reports from outputs/rankings/ into public/ and
generates an index.html with the newest report embedded first.
"""

import argparse
import os
import re
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish-dir", default="public")
    args = ap.parse_args()

    src = "outputs/rankings"
    pub = args.publish_dir
    os.makedirs(pub, exist_ok=True)

    reports = sorted(
        (f for f in os.listdir(src) if re.match(r"ranking_\d{4}-\d{2}-\d{2}\.html$", f)),
        reverse=True,
    )
    if not reports:
        raise SystemExit("No ranking reports found in outputs/rankings/")

    for f in os.listdir(src):
        if f.endswith((".html", ".json")):
            shutil.copy2(os.path.join(src, f), os.path.join(pub, f))

    items = "\n".join(
        f'<li><a href="{f}">{f[8:-5]}</a></li>' for f in reports
    )
    latest = reports[0]
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gauravi — NIFTY-100 Weekly Rankings</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:30px 16px;line-height:1.5}}
.wrap{{max-width:960px;margin:0 auto}}
h1{{font-size:1.5rem}}h2{{color:#58a6ff;font-size:1.05rem;margin-top:26px}}
.sub{{color:#8b949e;margin-bottom:14px}}
iframe{{width:100%;height:1200px;border:1px solid #30363d;border-radius:10px;background:#0d1117}}
li{{margin:3px 0}}a{{color:#58a6ff}}
.disc{{color:#8b949e;font-size:.75rem;margin-top:20px}}
</style></head><body><div class="wrap">
<h1>NIFTY-100 Weekly Rankings</h1>
<div class="sub">momentum + reversal blend &middot; updated every Monday</div>
<h2>Latest report ({latest[8:-5]})</h2>
<iframe src="{latest}" title="latest ranking"></iframe>
<h2>Archive</h2>
<ul>{items}</ul>
<p class="disc">Research and educational purposes only. Not investment advice.</p>
</div></body></html>"""
    with open(os.path.join(pub, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Published {len(reports)} report(s) to {pub}/ (latest: {latest})")


if __name__ == "__main__":
    main()
