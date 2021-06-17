This is an unofficial Python implementation of Hearing-Aid Speech Perception Index (HASPI)

Usage:
    from pyhaspi2 import haspi_v2
    haspi(x,fx,y,fy, HL = np.zeros(6))
    haspi_v2(x, fx, y, fy, HL = np.zeros(6))  

    Where x is clean reference speech, fx is sampling rate of x, and y is degraded speech, and fy is sampling rate of y. HL is 6-dim vectors represents hearing loss at the 6 audiometric frequencies, with 0 normal.

Differences between the original Matlab implementation by Prof. James M.Kates:
    1). In our implementation, we did not apply built-in alignment, so please make sure x and y are already aligned before using this function
    2). In our implementation of HASPI version 2, we use sigmoid weight approximation model shown in Eq.(1) in Ref.[2]instead of neural network weights
    3). We internally normalized both x and y into rms=1, and reference SPL level was set to 65 dB
    4). We only confirmed output results are correct in the case of HL=[0,0,0,0,0,0] (as normal hearing condition). Although HL vector can be set arbitrary in the function, we cannot guarantee results are correct since lack of extensive test.

Reference:
    [1]. Kates, James M., and Kathryn H. Arehart. "The Hearing-Aid Speech Perception Index (HASPI)." Speech Communication 65 (2014): 75-93.
    [2]. Kates, James M., and Kathryn H. Arehart. "The Hearing-Aid Speech Perception Index (HASPI) Version 2." Speech Communication (2020).
