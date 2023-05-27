import os
import re

folder = 'D:\\test'


def rec_replace(folder, c_from, c_to):
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            path = os.path.join(root, filename)
            newname = re.sub(r'[\s_]+', ' ', filename)
            newname = newname.strip()
            newpath = os.path.join(root,newname)
            os.rename(path, newpath)


if __name__ == '__main__':
    rec_replace(folder, '_', ' ')
