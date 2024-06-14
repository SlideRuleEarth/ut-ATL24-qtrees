#!/usr/bin/env python3
"""
Generate bash commands for cross validation
"""

import argparse
import sys
import glob


def main(args):

    # Show args
    if args.verbose:
        print(args, file=sys.stderr)

    # Get the filenames
    filenames = glob.glob (args.input_glob)
    print(f'{len(filenames)} total files', file=sys.stderr)

    # bash header
    print('#!/usr/bin/bash')
    print('')
    print('# Bash strict mode')
    print('set -euo pipefail')
    print('IFS=$\'\\n\\t\'')
    print('')
    print('# Create a temp directory')
    print('trap cleanup SIGINT SIGTERM ERR EXIT')
    print('tmpdir=$(mktemp --tmpdir --directory tmp.XXXXXXXX)')
    print('')
    print('function cleanup {')
    print('    echo "Cleaning up..."')
    print('    trap - SIGINT SIGTERM ERR EXIT')
    print('    rm -rf ${tmpdir}')
    print('}')
    print('')
    print('# Create a subdirectory for each split')
    for i in range(args.splits):
        print(f'mkdir ${{tmpdir}}/{i}/')
    print('')
    print('# Copy files to each subdirectory')
    for i, fn in enumerate(filenames):
        split = i % args.splits
        print(f'cp {fn} ${{tmpdir}}/{split}/')
    print('# Run train/classify/score')
    for i in range(args.splits):
        split = i % args.splits
        nonsplit_string = ','.join([str(x) for x in range(args.splits)
                                   if x != split])
        print(f'make INPUT="${{tmpdir}}/{{{nonsplit_string}}}/*.csv"'
              f' BUILD={args.build} '
              f' MODEL=${{tmpdir}}/{split}/model.json'
              f' train')
        print(f'make INPUT="${{tmpdir}}/{split}/*.csv"'
              f' BUILD={args.build} '
              f' MODEL=${{tmpdir}}/{split}/model.json'
              f' classify')
        print(f'make --no-print-directory INPUT="${{tmpdir}}/{split}/*.csv"'
              f' BUILD={args.build} '
              f' MODEL=${{tmpdir}}/{split}/model.json'
              f' score > cross_val.{split}.txt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Cross validation command generator')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show verbose output')
    parser.add_argument(
        '-s', '--splits', type=int, default=5,
        help='Number of cross-val splits')
    parser.add_argument(
        '-b', '--build', type=str, default="debug",
        help='Build type [debug|release]')
    parser.add_argument(
        'input_glob',
        type=str,
        help='Input training filename glob')

    args = parser.parse_args()

    main(args)
