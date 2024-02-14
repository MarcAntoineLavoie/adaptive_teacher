import os

base_dir = './output'
assert os.path.isdir(base_dir)
dirs = next(os.walk(base_dir))[1]
long_ckpts = ['model_0049999.pth', 'model_0079999.pth', 'model_0094999.pth', 'model_final.pth']
short_ckpts = ['model_0034999.pth', 'model_final.pth']
for dir in dirs:
    curr_dir = '/'.join((base_dir, dir))
    files = next(os.walk(curr_dir))[2]
    ckpts = [x for x in files if 'model' in x]
    if 'short' in dir:
        purge_list = [x for x in ckpts if x not in short_ckpts]
    else:
        purge_list = [x for x in ckpts if x not in long_ckpts]
    purge_names = ['/'.join((curr_dir, x)) for x in purge_list]

    for name in purge_names:
        if os.path.exists(name):
            # print(name)
            os.remove(name)

# a = 1