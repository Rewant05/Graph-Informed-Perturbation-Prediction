"""Named raw-count entry point; legacy script is preserved in archive/."""
from preprocessing import main
import sys

if __name__ == '__main__':
    raise SystemExit(main(['raw', *sys.argv[1:]]))
