import json
import shutil
import argparse
import os

if __name__ == '__main__':
    # parse arguments:
    # required argument - path to image or directory with images
    # optional argument - image extensions to classify (default: jpg;jpeg;png, used only if path is a directory)
    # optional argument - name of file to save results
    # optional argument - directory to copy classified images to subdirectories by class
    #                     (if not specified, images will not be copied)
    # optional argument - whether to clear output directory before copying
    # optional argument - score threshold to copy images (by default 0.5, used only if copy_images is True)
    # one optional - path to model config in json format (default: model.json)

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='path to image')
    parser.add_argument('--ext', type=str, default='jpg;jpeg;png',
                        help='image extensions to classify separated by `;` (default: jpg;jpeg;png)')
    parser.add_argument('--save', type=str, help='path to save results (json or yaml)')
    parser.add_argument('--out_dir', type=str, help='path to directory to copy classified images to')
    parser.add_argument('--clear_out_dir', action='store_true', help='clear output directory before copying')
    parser.add_argument('--threshold', type=float, default=0.5, help='score threshold to copy images (default: 0.5)')
    parser.add_argument('--model_config', type=str, default='model.json', help='path to model config in json format')
    args = parser.parse_args()

    # load model config
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)
    bk = model_config.get('backend', 'tf')
    if bk == 'tf':
        from tf_funcs import classify_image, classify_all_in_directory
    elif bk == 'torch':
        from torch_funcs import classify_image, classify_all_in_directory
    else:
        raise ValueError(f'Unknown backend: {bk}')

    if not os.path.isdir(args.image_path):
        # classify image
        if not os.path.exists(args.image_path):
            print(f'File {args.image_path} does not exist')
        else:
            object_name, score = classify_image(model_config['model_path'], model_config['classes'], args.image_path)
            print(f'Object: {object_name}, score: {score}')
            if args.save:
                if args.save.endswith('.json'):
                    with open(args.save, 'w') as f:
                        json.dump({args.image_path: (object_name, score)}, f)
                elif args.save.endswith('.yaml') or args.save.endswith('.yml'):
                    import yaml
                    with open(args.save, 'w') as f:
                        yaml.dump({args.image_path: (object_name, score)}, f)
                else:
                    print(f'Unknown file extension {args.save}, results will be saved in json format')
                    with open(args.save, 'w') as f:
                        json.dump({args.image_path: (object_name, score)}, f)
    else:
        # classify all images in directory
        results = classify_all_in_directory(model_config['model_path'], model_config['classes'], args.image_path)
        if args.save:
            if args.save.endswith('.json'):
                with open(args.save, 'w') as f:
                    json.dump(results, f)
            elif args.save.endswith('.yaml') or args.save.endswith('.yml'):
                import yaml
                with open(args.save, 'w') as f:
                    yaml.dump(results, f)
            else:
                print(f'Unknown file extension {args.save}, results will be saved in json format')
                with open(args.save, 'w') as f:
                    json.dump(results, f)
        else:
            for image, result in results.items():
                print(f'{image}: {result[0]}, score: {result[1]}')

        if args.out_dir:
            if args.clear_out_dir:
                shutil.rmtree(args.out_dir, ignore_errors=True)
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir, exist_ok=True)

            print(f'Copying images to {args.out_dir} by class...')
            # print progress bar
            from tqdm import tqdm

            # copy images to subdirectories by class
            for image, (object_name, score) in tqdm(results.items(), total=len(results), desc='Copying images'):
                if score >= args.threshold:
                    os.makedirs(os.path.join(args.out_dir, object_name), exist_ok=True)
                    shutil.copy(os.path.join(args.image_path, image), os.path.join(args.out_dir, object_name, image))
