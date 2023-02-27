
from glob import glob
import itertools

def get_name(filepath):
    name = filepath.replace('\\','/').split('/')[-1][:-4]
    if name[:-2].endswith('attractor'):
        name = name[:-2]
    return name
    
def get_dir(filepath):
    return '/'.join(filepath.replace('\\','/').split('/')[:-1])

def get_system(name):
    return name.split('-')[0]

def main(filename):
    with open(filename, 'w') as file:
        filenames = list(map(lambda x:x.replace('\\','/'), 
                itertools.chain(glob('histograms/*.png'), glob('histograms/traintime/*.png'))))
        
        names = [get_name(x) for x in filenames]
        dirs = [get_dir(x) for x in filenames]
        systems = [get_system(x) for x in names]
        
        last_name = ''
        last_system = ''
        
        for name, system, cdir, filename in zip(names, systems, dirs, filenames):
            if system != last_system:
                file.write(f'<h1>{system.capitalize()}</h1>\n')
            if name != last_name:
                file.write(f'<h2>{name.replace("-"," ")}</h2>\n')
            if name.endswith('attractor'):
                pass 
                #file.write(filename+'\n')
            
            if cdir == 'attractor':
                size = 'width="400" height="320"'
            elif cdir.startswith('histograms'):
                size = 'width="400" height="530"'
            else:
                size = 'width="500" height="400"'
                
            if filename.lower().endswith('.pdf'):
                file.write(f'<embed src="./{filename}?#scrollbar=0&toolbar=0&navpanes=0" {size} type="application/pdf">\n')
            elif filename.lower().endswith('.png'):
                file.write(f'<img src="./{filename}" {size}>\n')
            else:
                print(f"Unsure what to do with {filename}")
            last_name, last_system = name,system
        
    

if __name__=="__main__":
    main('hist-plots.html')