'''

@File    :   test.py 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-07-05 13:18   shenhangke      1.0         None
---------------------
 
'''

import ctypes

ll = ctypes.cdll.LoadLibrary

lib = ll("./libinit.dylib")
lib.setInPath("./data/FB15K/".encode("ascii"))
lib.setBernFlag(0)

lib.init()
print(lib.getRelationTotal())
print("process finish")
