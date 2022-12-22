from tqdm import tqdm

import ann_automl.core.db_module as db
import argparse
import os
import sys


def init_db(dbfile, dbconfig):
    newdb = db.DBModule(dbstring=dbfile, dbconf_file=dbconfig)  # datasets.sqlite - название файла с корпусом аннотаций
    newdb.fill_all_default()


if __name__ == '__main__':
    # arguments:
    #   --out (-o): database file name (default: datasets.sqlite), this file will be created
    #   --force: if the database file already exists, it will be overwritten
    #   --conf: database configuration file (default: dbconfig.json)

    parser = argparse.ArgumentParser(description='Database initialization script')
    parser.add_argument('--out', '-o', default='datasets.sqlite', help='database file name')
    parser.add_argument('--force', action='store_true', help='overwrite existing database file')
    parser.add_argument('--conf', default='dbconfig.json', help='database configuration file')
    args = parser.parse_args()

    backup = None
    try:
        if os.path.isfile(args.out):
            if not os.path.isfile(args.dbconf):
                print(f'File {args.dbconf} not found', file=sys.stderr)
                sys.exit(1)
            if args.force:
                # backup existing database file
                backup = args.out + '.bak'
                os.rename(args.out, backup)
            else:
                print(f'File {args.out} already exists, use --force to overwrite it', file=sys.stderr)
                sys.exit(1)
        init_db(args.out, args.dbconf)
        if backup is not None:
            os.remove(backup)
    except Exception as e:
        if backup is not None:
            os.rename(backup, args.out)
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
