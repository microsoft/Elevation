import json, requests
import numpy as np
import elevation.load_data
import time
import pandas

if __name__ == "__main__":
    # url = 'http://elevation.southcentralus.cloudapp.azure.com:5000/elevation'
    url = 'http://127.0.0.1:5000/elevation'

    # this is from the mouse data, first two rows
    guides = [['GGTCTGAGTCGGGGCCAGGGCGG', 'GGTCTGAGTCGGAGCCAGGGCGG', 0.99291331],
              ['CGGGAAACACAGAAAGCCAAGGG', 'CGGGAGACACAGAAAGCCAAGGG', 0.61870314]]

    resp = requests.post(url=url, data={'wildtype': (guides[0][0], guides[1][0]), 'offtarget': (guides[0][1], guides[1][1])})
    data = json.loads(resp.text)
    print data['elevation score']
    assert np.allclose(data['elevation score'], np.array(np.array(guides)[:, 2],dtype=float))

    for g in guides:
        data = {'wildtype': g[0],'offtarget': g[1]}

        resp = requests.post(url=url, data=data)
        data = json.loads(resp.text)
        assert np.allclose(data['elevation score'], g[2], atol=1e-5)


    print "queried %d guides, all good" % len(guides)

    mouse_data = elevation.load_data.load_mouse_data()[0]

    # start = time.time()
    # for i in range(mouse_data.shape[0]):
    #     if (i+1) % 10 == 0:
    #         print "Timing experiment: %d guides scored in %f seconds" % (i+1, time.time()-start)
    #     resp = requests.post(url=url, data={'wildtype':mouse_data.iloc[i]['WTSequence'],
    #                                         'offtarget': mouse_data.iloc[i]['MutatedSequence']})

    start = time.time()
    data={'wildtype':mouse_data['WTSequence'].tolist(), 'offtarget':mouse_data['MutatedSequence'].tolist()}
    resp = requests.post(url=url, data=data)
    data = json.loads(resp.text)
    print "Batch timing experiment: %d guides scored in %f seconds in one single batch" % (mouse_data.shape[0], time.time()-start)


    hsu_data = elevation.load_data.load_HsuZang_data(version='hsu-zhang-single')[0]
    start = time.time()
    data={'wildtype':hsu_data['WTSequence'].tolist(), 'offtarget':hsu_data['MutatedSequence'].tolist()}
    resp = requests.post(url=url, data=data)
    data = json.loads(resp.text)
    print "Batch timing experiment: %d guides scored in %f seconds in one single batch" % (hsu_data.shape[0], time.time()-start)


    mega_hsu_data =  pandas.concat((hsu_data, hsu_data, hsu_data, hsu_data, hsu_data, hsu_data, hsu_data, hsu_data, hsu_data))
    start = time.time()
    data={'wildtype':mega_hsu_data['WTSequence'].tolist(), 'offtarget':mega_hsu_data['MutatedSequence'].tolist()}
    resp = requests.post(url=url, data=data)
    data = json.loads(resp.text)
    print "Batch timing experiment: %d guides scored in %f seconds in one single batch" % (mega_hsu_data.shape[0], time.time()-start)
