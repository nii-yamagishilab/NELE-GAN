This is an unofficial Python implementation of Hearing-Aid Speech Perception Index (HASPI) proposed by Prof. James M. Kates, and this unofficial Python implementation was redistributed with the permission of Professor James M. Kates.

Usage:
    from pyhaspi2 import haspi_v2
    haspi(x,fx,y,fy, HL = np.zeros(6))
    haspi_v2(x, fx, y, fy, HL = np.zeros(6))  

    Where x is clean reference speech, fx is sampling rate of x; and y is degraded speech, fy is sampling rate of y. HL is 6-dim vectors represents hearing loss at the 6 audiometric frequencies.

Differences between the original Matlab implementation written by Prof. James M. Kates and this Python implementation:
    1). In our implementation, we did not apply built-in alignment, so please make sure x and y are already aligned before using the functions
    2). In our implementation of HASPI version 2, we use sigmoid weight approximation model shown in Eq.(1) in Ref.[2]instead of neural network weights
    3). We internally normalized both x and y into rms=1, and reference SPL level was set to (constant) 65 dB.
    4). We can only confirm output results are correct in the case of normal hearing condition, i.e., HL=[0,0,0,0,0,0]. Although HL vector can be set arbitrary in the function, we cannot guarantee results are correct since no extensive test done.

Reference:
    [1]. Kates, James M., and Kathryn H. Arehart. "The Hearing-Aid Speech Perception Index (HASPI)." Speech Communication 65 (2014): 75-93.
    [2]. Kates, James M., and Kathryn H. Arehart. "The Hearing-Aid Speech Perception Index (HASPI) Version 2." Speech Communication (2020).
