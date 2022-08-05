import os
import shutil
import subprocess
import random
import string
import numpy as np
import time, datetime
import zmq, json
import sys


import socket
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=6666, required=False)
args = parser.parse_args()

print("server is running at %s@%s" % (args.port, socket.gethostname()))


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % (args.port))


def feff_calculator_mpi(js):
    """
    # USAGE:
    with open('/data/scratch/zmq_servers/FEFF/test/feff.inp') as fi:
        feffinp = fi.readlines()
    js =  {'task':'feff_run','feff_inp': feffinp,'ncores':12}
    js_out = feff_calculator_mpi(js)
    """

    # runtime parameters
    try:
        ncores = js["ncores"]
    except:
        ncores = 4

    try:
        feff_scratch_path = js["feff_scratch_path"]
    except:
        feff_scratch_path = "/data/scratch/zmq_servers/FEFF/"

    try:
        mpirun_cmd = js["mpirun_cmd"]
    except:
        mpirun_cmd = "/opt/intel/oneapi/mpi/2021.5.1/bin/mpirun"

    try:
        exe_path = js["exe_path"]
    except:
        exe_path = "/data/software/FEFF/bin/mpi/"

    try:
        cleanup = js["cleanup"]
    except:
        cleanup = "false"

    try:

        os.chdir(feff_scratch_path)
        uid = "feff_" + "".join(
            random.choices(string.ascii_uppercase + string.ascii_lowercase, k=10)
        )
        os.makedirs(uid, exist_ok=True)
        os.chdir(uid)

        print(
            "Running FEFF calculation at \n %s/%s \n using %d cores\n"
            % (feff_scratch_path, uid, ncores)
        )

        fi = open("feff.inp", "a")
        for ii in js["feff_inp"]:
            fi.write(ii)
        fi.close()

        try:
            js["cif"]
            cif = open("structure.cif", "a")
            for cc in js["cif"]:
                cif.write(cc)
            cif.close()
        except:
            pass

        exe_list = [
            "rdinp",
            "dmdw",
            "atomic",
            "pot",
            "ldos",
            "screen",
            "crpa",
            "opconsat",
            "xsph",
            "fms",
            "mkgtr",
            "path",
            "genfmt",
            "ff2x",
            "sfconv",
            "compton",
            "eels",
            "rhorrp",
        ]

        start_time = time.time()

        open("feff.out", "a").close()
        for e in exe_list:
            _ = subprocess.run(
                ["%s -np %d %s/%s >> feff.out" % (mpirun_cmd, ncores, exe_path, e)],
                shell=True,
            )

            if _.returncode > 0:
                print("error at %s" % e)
                print(_)
                break
        finish_time = time.time()

        with open("feff.inp") as f:
            feffinp = f.readlines()
        with open("feff.out") as f:
            feffout = f.readlines()

        if os.path.isfile("xmu.dat"):
            with open("xmu.dat") as f:
                xmudat = f.readlines()
            endhindex = xmudat.index("#  omega    e    k    mu    mu0     chi     @#\n")
            js = {
                "fname": "xmu.dat",
                "header": xmudat[0 : endhindex + 1],
                "omega": [float(i.split()[0]) for i in xmudat[endhindex + 1 :]],
                "e": [float(i.split()[1]) for i in xmudat[endhindex + 1 :]],
                "mu": [float(i.split()[3]) for i in xmudat[endhindex + 1 :]],
                "mu0": [float(i.split()[4]) for i in xmudat[endhindex + 1 :]],
                "chi": [float(i.split()[5]) for i in xmudat[endhindex + 1 :]],
                "start_time": datetime.datetime.fromtimestamp(start_time).strftime(
                    "%Y-%m-%d__%H:%M:%S"
                ),
                "finish_time": datetime.datetime.fromtimestamp(finish_time).strftime(
                    "%Y-%m-%d__%H:%M:%S"
                ),
                "time_elapsed": finish_time - start_time,
                "feffinp": feffinp,
                "feffout": feffout,
                "mpirun_cmd": mpirun_cmd,
                "ncores": ncores,
                "exe_path": exe_path,
            }
        else:
            js = {
                "start_time": datetime.datetime.fromtimestamp(start_time).strftime(
                    "%Y-%m-%d__%H:%M:%S"
                ),
                "finish_time": datetime.datetime.fromtimestamp(finish_time).strftime(
                    "%Y-%m-%d__%H:%M:%S"
                ),
                "time_elapsed": finish_time - start_time,
                "feffinp": feffinp,
                "feffout": feffout,
                "mpirun_cmd": mpirun_cmd,
                "ncores": ncores,
                "exe_path": exe_path,
            }

        os.chdir("..")
        if cleanup == "true":
            shutil.rmtree(uid)

        return js

    except Exception as exc:
        print("something is wrong")
        print(exc)




def fdmnes_calculator_mpi(js):
    '''
    # USAGE:
    with open('/data/software/FDMNES/fdmnes_tests/4_nospin/fdmnes.inp') as fi:
        fdmnesinp = fi.readlines()
    js =  {'task':'fdmnes_run', 'fdmnes_inp': fdmnesinp, 'ncores':12} 
    js_out = fdmnes_calculator_mpi(js)
    js_out
    '''
    
    #runtime parameters
    try:
        ncores = js['ncores']
    except:
        ncores=4

        
    try:
        fdmnes_scratch_path = js['fdmnes_scratch_path']
    except:
        fdmnes_scratch_path = '/data/scratch/zmq_servers/FDMNES/'        
        
    try:
        mpirun_cmd = js['mpirun_cmd']
    except:
        mpirun_cmd = '/opt/intel/oneapi/mpi/2021.5.1/bin/mpirun' 
    
    try:
        exe_path = js['exe_path']
    except:
        exe_path = '/data/software/FDMNES/parallel_fdmnes'

    try:
        mumps = js["HOST_NUM_FOR_MUMPS"]
    except:
        mumps = 1
        
    try:
        cleanup = js['cleanup']
    except:
        cleanup = 'false'
        
    
    
    
    try:

        os.chdir(fdmnes_scratch_path)
        uid = 'fdmnes_'+''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=10))
        os.makedirs(uid,exist_ok=True)
        os.chdir(uid)
        
        print('Running FDMNES calculation at \n %s/%s \n using %d cores\n'%(fdmnes_scratch_path,uid,ncores))
        
        fi = open("fdmnes.inp", "a")
        for ii in js['fdmnes_inp']:
            fi.write(ii)
        fi.close()
        

        try:
            js['cif']
            cif = open("structure.cif", "a")
            for cc in js['cif']:
                cif.write(cc)
            cif.close()
        except:
            pass




        #fdmnes wants this
        fi = open("fdmfile.txt", "a")
        fi.write('1\n')
        fi.write('fdmnes.inp\n')
        fi.close()          
        
        exe_list = [
            'fdmnes_mpi_linux64',
        ]

        start_time = time.time()  

        open('fdmnes.out', 'a' ).close()

        for e in exe_list:
            _ = subprocess.run(
                [
                    "HOST_NUM_FOR_MUMPS=%s %s -np %d %s/%s >> fdmnes.out"
                    % (mumps, mpirun_cmd, ncores, exe_path, e)
                ],
                shell=True,
            )

            if _.returncode > 0:
                print('error at %s'%e)
                print(_)
                break
        finish_time = time.time()
        

        
      

        with open('fdmnes.inp') as f:
            fdmnesinp = f.readlines()
        with open('fdmnes.out') as f:
            fdmnesout = f.readlines()
        with open('fdmnes_bav.txt') as f:
            fdmnesbav = f.readlines()

        js = {
            'start_time': datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d__%H:%M:%S'),
            'finish_time': datetime.datetime.fromtimestamp(finish_time).strftime('%Y-%m-%d__%H:%M:%S'),
            'time_elapsed': finish_time - start_time,
            'fdmnesinp': fdmnesinp,
            'fdmnesout': fdmnesout,
            'fdmnesbav': fdmnesbav,
            'mpirun_cmd': mpirun_cmd,
            'ncores': ncores,
            'exe_path': exe_path,
            }            
            


        # single site
        if os.path.isfile('fdmnes.txt'):
            with open('fdmnes.txt') as f:
                fdmnestxt = f.readlines()
            # non-magnetic calculation
            if len(fdmnestxt[-1].split()) == 2:
                js_new = {
                    'fname_fdmnestxt': 'fdmnes.txt',
                    'header_fdmnestxt': fdmnestxt[0:2],
                    'e':     [float(i.split()[0]) for i in fdmnestxt[2:]],
                    'mu':    [float(i.split()[1]) for i in fdmnestxt[2:]],
                    }
                js.update(js_new)
            # magnetic calculation
            elif len(fdmnestxt[-1].split()) == 3:
                js_new = {
                    'fname_fdmnestxt': 'fdmnes.txt',
                    'header_fdmnestxt': fdmnestxt[0:2],
                    'e':     [float(i.split()[0]) for i in fdmnestxt[2:]],
                    'mu_up':  [float(i.split()[1]) for i in fdmnestxt[2:]],
                    'mu_dw':  [float(i.split()[2]) for i in fdmnestxt[2:]],
                    'mu':  [(float(i.split()[1])+float(i.split()[2])) for i in fdmnestxt[2:]],
                    }
                js.update(js_new)


        # multiple sites
        else:

            import glob 
            outputs = glob.glob('fdmnes_*.txt')

            for i in outputs:
                try:
                    t = int(i.split('_')[1].split('.txt')[0])
                    
                    with open('fdmnes_%d.txt'%t) as f:
                        fdmnestxt = f.readlines()
                            
                    # non-magnetic calculation
                    if len(fdmnestxt[-1].split()) == 2:
                        js_new = {
                            'fname_fdmnestxt_%d'%t: i,
                            'header_fdmnestxt_%d'%t: fdmnestxt[0:2],
                            'e_%d'%t:     [float(i.split()[0]) for i in fdmnestxt[2:]],
                            'mu_%d'%t:    [float(i.split()[1]) for i in fdmnestxt[2:]],
                            }
                        js.update(js_new)

                    # magnetic calculation
                    elif len(fdmnestxt[-1].split()) == 3:
                        js_new = {
                            'fname_fdmnestxt_%d'%t: i,
                            'header_fdmnestxt_%d'%t: fdmnestxt[0:2],
                            'e_%d'%t:     [float(i.split()[0]) for i in fdmnestxt[2:]],
                            'mu_up_%d'%t:  [float(i.split()[1]) for i in fdmnestxt[2:]],
                            'mu_dw_%d'%t:  [float(i.split()[2]) for i in fdmnestxt[2:]],
                            'mu_%d'%t:  [(float(i.split()[1])+float(i.split()[2])) for i in fdmnestxt[2:]],
                            }
                        js.update(js_new)  

                except:
                    pass            





            
        if os.path.isfile('fdmnes_conv.txt'):
            with open('fdmnes_conv.txt') as f:
                fdmnestxt_conv = f.readlines()
            js_new = {
                'fname_fdmnesconvtxt': 'fdmnes_conv.txt',
                'header_fdmnesconvtxt': fdmnestxt[0:1],
                'e_conv':     [float(i.split()[0]) for i in fdmnestxt_conv[1:]],
                'mu_conv':    [float(i.split()[1]) for i in fdmnestxt_conv[1:]],
                }
            js.update(js_new)


        os.chdir('..')

        if cleanup == 'true':
            shutil.rmtree(uid)
            
            
        return js


    except Exception as exc:
        print('something is wrong')
        print(exc)





while True:
    #     clear_output()

    try:
        js_in = json.loads(socket.recv_string())
        print("\n")
        print(js_in["task"])

        if js_in["task"] == "server_test":
            js_out = {"status_reply": "server is live and running at localhost:6666"}
            socket.send_string(json.dumps(js_out, indent=4))

        elif js_in["task"] == "feff_run":
            js_out = feff_calculator_mpi(js_in)
            socket.send_string(json.dumps(js_out, indent=4))

        elif js_in["task"] == "fdmnes_run":
            js_out = fdmnes_calculator_mpi(js_in)
            socket.send_string(json.dumps(js_out, indent=4))

    except Exception as exc:
        socket.send_string("ERROR!!!!\n%s\n" % (exc), encoding="utf-8")
        print(exc)
