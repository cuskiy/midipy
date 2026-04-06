"""Entry point. See README.md for architecture."""

VERSION = "v137"


def main():
    import argparse
    from midi import parse
    from mix import render

    ap = argparse.ArgumentParser(description="Render MIDI to FLAC")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--spatial", action="store_true")
    ap.add_argument("--stems", action="store_true",
                    help="Export per-instrument stem files alongside the mix")
    ap.add_argument("--stems-dir", default=None,
                    help="Custom output directory for stems (default: <output>_stems/)")
    ap.add_argument("--start", type=float, default=None,
                    help="Start time in seconds")
    ap.add_argument("--end", type=float, default=None,
                    help="End time in seconds")
    a = ap.parse_args()
    print(VERSION)
    tracks, pans, ch_data = parse(a.input)
    out = a.output or a.input.rsplit(".", 1)[0] + ".flac"
    render(tracks, pans, out, spatial=a.spatial, ch_data=ch_data,
           stems=a.stems, stems_dir=a.stems_dir, region=(a.start, a.end))


if __name__ == "__main__":
    main()
