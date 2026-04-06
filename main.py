"""Entry point. See README.md for architecture."""

VERSION = "v149"


def main() -> None:
    import argparse, sys, os
    from midi import parse
    from mix import render

    ap = argparse.ArgumentParser(description="Render MIDI to FLAC")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--spatial", action="store_true")
    ap.add_argument("--stems", action="store_true",
                    help="Export per-instrument stem files alongside the mix")
    ap.add_argument("--stems-dir", default=None,
                    help="Custom output directory for stems (default: <o>_stems/)")
    ap.add_argument("--start", type=float, default=None,
                    help="Start time in seconds")
    ap.add_argument("--end", type=float, default=None,
                    help="End time in seconds")
    a = ap.parse_args()
    print(VERSION)

    if not os.path.isfile(a.input):
        print(f"Error: file not found: {a.input}", file=sys.stderr)
        sys.exit(1)

    try:
        result = parse(a.input)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    out = a.output or a.input.rsplit(".", 1)[0] + ".flac"
    render(result, out, spatial=a.spatial, stems=a.stems,
           stems_dir=a.stems_dir, region=(a.start, a.end))


if __name__ == "__main__":
    main()
