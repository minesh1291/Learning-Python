```
ipython notebook --ip=0.0.0.0 --port=80

usage: ipython [-h] [--certfile NOTEBOOKAPP.CERTFILE] [--ip NOTEBOOKAPP.IP]
               [--pylab [NOTEBOOKAPP.PYLAB]]
               [--log-level NOTEBOOKAPP.LOG_LEVEL]
               [--port-retries NOTEBOOKAPP.PORT_RETRIES]
               [--notebook-dir NOTEBOOKAPP.NOTEBOOK_DIR]
               [--config NOTEBOOKAPP.CONFIG_FILE]
               [--keyfile NOTEBOOKAPP.KEYFILE] [--port NOTEBOOKAPP.PORT]
               [--transport KERNELMANAGER.TRANSPORT]
               [--browser NOTEBOOKAPP.BROWSER] [--script] [-y] [--no-browser]
               [--debug] [--no-mathjax] [--no-script] [--generate-config]

ps auxww | grep 'ipython' | awk '{print $2}' | xargs sudo kill -9
```
