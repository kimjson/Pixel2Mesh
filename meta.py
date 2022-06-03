base_path = '/Users/jaeseong/Library/Mobile Documents/com~apple~CloudDocs/KAIST/2022s/CS492/Project/Pixel2Mesh/data/meta'
meta_file_path = '/Users/jaeseong/Library/Mobile Documents/com~apple~CloudDocs/KAIST/2022s/CS492/Project/Pixel2Mesh/data/meta/train_list.txt'

with open(meta_file_path, 'r') as meta_file:
    # Data/ShapeNetP2M/04256520/1a4a8592046253ab5ff61a3a2a0e2484/rendering/00.dat
    paths_by_category = dict()
    paths_by_model = dict()

    for file_path in meta_file:
        paths = file_path.split('/')
        category_id = paths[2]
        model_id = paths[3]

        paths_in_category = paths_by_category.get(category_id, [])
        paths_by_category[category_id] = paths_in_category + [file_path]

        paths_in_model = paths_by_model.get(model_id, [])
        paths_by_model[model_id] = paths_in_model + [file_path]


    for category_id, files in paths_by_category.items():
        with open(f'{base_path}/train_list_{category_id}.txt', 'w') as category_meta_file:
            category_meta_file.writelines(files)


    print(f'Number of models: {len(paths_by_model)}')
            