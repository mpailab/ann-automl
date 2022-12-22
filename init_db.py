from tqdm import tqdm

import ann_automl.core.db_module as db
import argparse
import os
import sys


def init_db(dbfile, dbconfig):
    newdb = db.DBModule(dbstring=dbfile, dbconf_file=dbconfig)  # datasets.sqlite - название файла с корпусом аннотаций
    newdb.fill_all_default()
    return newdb


if __name__ == '__main__':
    # arguments:
    #   --db (-o): database file name (default: datasets.sqlite), this file will be created
    #   --init: initialize database and fill with datasets
    #           Kaggle cats vs dogs, ImageNet, COCO (that must be downloaded manually)
    #   --force: if the database file already exists, it will be overwritten
    #   --conf: database configuration file (default: dbconfig.json)
    #   --add_tfds: list of tensorflow datasets to add to the database (separated by spaces)
    #   --add_dir_ds: add directory which contain subdirectories with images as dataset
    #   --name: name of dataset to add (if --add_tfds is specified, then default is the name of dataset in tensorflow)
    #           can only be specified if only one dataset is added
    #   --tfds_dir: directory where tensorflow datasets are stored (default: datasets/.tensorflow_datasets)

    parser = argparse.ArgumentParser(description='Database initialization script')
    parser.add_argument('--db', default='datasets.sqlite', help='sqlite database file name')
    parser.add_argument('--init', action='store_true',
                        help='initialize database and fill with datasets'
                             'Kaggle cats vs dogs, ImageNet, COCO (that must be downloaded manually)')
    parser.add_argument('--reset', '-r', action='store_true', help='delete existing database file if it exists')
    parser.add_argument('--conf', default='dbconfig.json', help='database configuration file')
    parser.add_argument('--add_tfds', nargs='*', help='list of tensorflow datasets to add to the database')
    parser.add_argument('--add_dir_ds', help='add directory which contain subdirectories with images as dataset')
    parser.add_argument('--name', '-n', help='name of dataset to add (if --add_tfds is specified, '
                                             'then default is the name of dataset in tensorflow)')

    args = parser.parse_args()
    backup = None
    try:
        if args.add_tfds:
            if args.add_dir_ds:
                print('Cannot simultaniously add both tensorflow datasets and directory. '
                      'Add them separately specifying --name argument for each dataset separately', file=sys.stderr)
                sys.exit(1)
            if len(args.add_tfds) > 1 and args.name:
                print('Cannot specify --name argument if more than one dataset is added. '
                      'Use --name argument for each dataset separately or don`t specify it (to use default names)',
                      file=sys.stderr)
                sys.exit(1)
        if args.add_dir_ds and not args.name:
            print('--name argument should be specified if subdurs dataset is added', file=sys.stderr)
            sys.exit(1)
        if os.path.isfile(args.db):
            if not os.path.isfile(args.dbconf):
                print(f'File {args.dbconf} not found', file=sys.stderr)
                sys.exit(1)
            if args.reset:
                # backup existing database file
                backup = args.db + '.bak'
                os.rename(args.db, backup)
            elif args.init:
                print(f'File {args.db} already exists, use --reset to overwrite it', file=sys.stderr)
                sys.exit(1)
        if args.init:
            newdb = init_db(args.db, args.dbconf)
        else:
            newdb = db.DBModule(dbstring=args.db, dbconf_file=args.dbconf)
            if not os.path.isfile(args.db):
                newdb.create_sqlite_file()
        if args.add_tfds:
            print('Adding tensorflow datasets to the database')
            for ds in args.add_tfds:
                print(f'Adding {ds}')
                try:
                    newdb.add_tensorflow_dataset(ds, args.name or None, ds_path=args.tfds_dir)
                except KeyboardInterrupt:
                    print('Interrupted')
                    break
                except Exception as e:
                    print(f'Error while adding {ds}: {e}')
        if args.add_dir_ds:
            print('Adding images from directory as a dataset to the database')
            try:
                newdb.add_dataset_from_subdirs(args.add_dir_ds, args.name)
            except KeyboardInterrupt:
                print('Interrupted')
            except Exception as e:
                print(f'Error while adding {args.add_dir_ds}: {e}')

        if backup is not None:
            os.remove(backup)
    except Exception as e:
        if backup is not None:
            os.rename(backup, args.db)
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
