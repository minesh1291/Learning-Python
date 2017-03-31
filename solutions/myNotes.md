### Python library
* Error
```
ImportError: No module named mpl_toolkits.mplot3d
```
* Solution
```bash
sudo apt-get install libfreetype6-dev
sudo pip install matplotlib
```
### Python scripting 
* Python script that prints its source
```py
with open(__file__) as f: print '\n'.join(f.read().split('\n')[1:])
```
