import os
import signal
import subprocess
import threading
class PopenTimeout(object):
    def __init__(self , cmd , input_file=None , output_file=None , shell=False):
        self.cmd = cmd
        self.input_file = input_file
        self.output_file = output_file
        self.process = None
        self.shell = shell
    def run(self, timeout, print_lvl=1):
        def target():
            #print self.cmd
            #print self.input_file
            #print self.output_file
            self.process = subprocess.Popen(self.cmd,stdin=self.input_file,stdout=self.output_file,shell=self.shell)
            self.process.communicate()
        if print_lvl == 1:
            print('Timeout:', timeout)
        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            if print_lvl == 1:
                print("Killing process after:",timeout,"s",self.process.pid)
            #try:
            #    os.killpg(self.process.pid, signal.SIGTERM)
            #except OSError:
            #    print 'OSERROR'
            self.process.kill()
            thread.join()
            return 130
        return self.process.returncode


class PopenStorePID(object):
    def __init__(self , cmd , input_file=None , output_file=None , shell=False):
        self.cmd = cmd
        self.input_file = input_file
        self.output_file = output_file
        self.process = None
        self.shell = shell
    def run(self, timeout, pid_dict, key, lock, print_lvl=0):
        import subprocess
        import threading
        def target():
            #print self.cmd
            #print self.input_file
            #print self.output_file
            self.process = subprocess.Popen(self.cmd,stdin=self.input_file,stdout=self.output_file,shell=self.shell)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()
        lock.acquire()
        pid_dict[key] = self.pid
        lock.release()
        thread.join(timeout)
        if thread.is_alive():
            if print_lvl == 1:
                print("Killing process after:",timeout,"s")
            self.process.kill()
            thread.join()
            return 130
        return self.process.returncode
