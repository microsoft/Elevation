import math
import os, pickle
import random
import copy
import time
import numpy as np
import settings

from warnings import warn


def get_pairwise_distance_mudra(r):
    pair_dis = np.abs(r - r[:, None])
    pair_dis = np.reshape(pair_dis,len(r)*len(r))
    pair_dis.sort()
    pair_dis = pair_dis[len(r):]
    return np.mean(pair_dis)


def get_gene_sequence(gene_name):
    try:
        gene_file = settings.seq_dir_template.format(gene_name=gene_name)
        with open(gene_file, 'rb') as f:
            seq = f.read()
            seq = seq.replace('\r', '')
            seq = seq.replace('\n', '')
    except:
        raise Exception("could not find gene sequence file %s, please see examples and generate one for your gene as needed, with this filename" % gene_file)
    return seq


def get_or_compute(file, fargpair, force_compute=False):
    try:
        if os.path.exists(file) and not force_compute:
            print "from get_or_compute reading cached pickle", file
            with open(file, 'rb') as f:
                return pickle.load(f)
        else:
            print "from get_or_compute recomputing pickle cache", file
    except Exception as e:
        # TODO: catch pickle failure error
        warn("Failed to load %s" % file)
        warn("Recomputing. This may take a while...")

    result = fargpair[0](*fargpair[1])
    with open(file, 'wb') as f:
        pickle.dump(result, f)
    return result


def to_temp(name, object):
    filename = settings.pj(settings.tmpdir, name + ".pkl")
    with open(filename, "wb") as f:
        pickle.dump(object, f)


def from_temp(name):
    filename = settings.pj(settings.tmpdir, name + ".pkl")
    with open(filename, "rb") as f:
        return pickle.load(f)


def execute_parallel(farg_pairs, num_procs=None, verbose=False):
    # see https://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/
    from multiprocess import Process, Queue, cpu_count

    if num_procs is None:
        # leave 25%
        num_procs = math.ceil(cpu_count()*.75)
        print "using %d procs in execute parallel" % num_procs

    processes = []
    q = None
    results = []
    q = Queue()

    num_jobs = len(farg_pairs)
    if verbose:
        print "execute_parallel num_procs=%d, num_jobs=%d" % (num_procs, num_jobs)

    i = -1
    farg_pair = None
    farg_pairs = copy.copy(farg_pairs)
    while len(farg_pairs) > 0:
        farg_pair = farg_pairs.pop(0)
        i += 1
        if verbose:
            print "running job", i

        def target_func(*args, **kwargs):
            q.put((i, farg_pair[0](*args, **kwargs)))

        if len(farg_pair) > 1:
            p = Process(target=target_func, args=farg_pair[1])
        else:
            p = Process(target=target_func)
        p.start()
        processes.append(p)

        # wait until we drop below num_procs
        while len(processes) >= num_procs:
            len1 = len(results)
            results.append(q.get())
            if len1 != len(results):
                for j, p in enumerate(processes):
                    if p.exitcode is not None:
                        p.join()
                        break
                processes = processes[:j] + processes[j+1:]
            else:
                time.sleep(0.01)

    while len(results) < num_jobs:
        results.append(q.get())
        time.sleep(0.01)

    assert len(results) == num_jobs

    # join remaining processes before exiting
    for i, p in enumerate(processes):
        p.join()

    results = zip(*sorted(results, key=lambda x: x[0]))[1]
    return results

if __name__ == "__main__":

    def test(t, v):
        time.sleep(random.randint(1, 4))
        return t, v

    commands = []
    for i in range(11):
        v = random.randint(0, 1000)
        commands.append((test, (i, v)))
    a = execute_parallel(commands, num_procs=None, verbose=True)
    b = map(lambda x: x[1], commands)
    print a
    print b
    assert tuple(a) == tuple(b)
