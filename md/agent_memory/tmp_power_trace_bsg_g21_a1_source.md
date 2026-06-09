target bp_processor/ic/node_1__io/io_async_io_resp_link/g21/A1 | AND2_X1 A1 input drv=0 | OR 1.77689715e+09/0.616040945 X 921871616/0.265844822 ratio 0.5188 net=bp_processor/ic/node_1__io/io_resp_link_lo[63]
driver candidate bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g212/ZN | NAND3_X1 ZN output drv=1 | OR 1.77689715e+09/0.616040945 X 921871616/0.265844822 ratio 0.5188 net=bp_processor/ic/node_1__io/io_resp_link_lo[63]
## depth 0: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g212 NAND3_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g212/A3 | NAND3_X1 A3 input drv=0 | OR 1.30435469e+09/0.683743894 X 109687232/0.981026351 ratio 0.08409 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_10
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g212/ZN | NAND3_X1 ZN output drv=1 | OR 1.77689715e+09/0.616040945 X 921871616/0.265844822 ratio 0.5188 net=bp_processor/ic/node_1__io/io_resp_link_lo[63]
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g212/A1 | NAND3_X1 A1 input drv=0 | OR 1.10121126e+09/0.724575043 X 525213728/0.851795554 ratio 0.4769 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_9
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g212/A2 | NAND3_X1 A2 input drv=0 | OR 930293696/0.775011361 X 463252640/0.878560781 ratio 0.498 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_11
choose input score=1.21319: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g212/A3
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g214/ZN | NAND2_X1 ZN output drv=1 | OR 1.30435469e+09/0.683743894 X 109687232/0.981026351 ratio 0.08409 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_10
## depth 1: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g214 NAND2_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g214/A1 | NAND2_X1 A1 input drv=0 | OR 1.21211533e+09/0.334770113 X 109635744/0.018981576 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__data_sel_lo[1]
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g214/ZN | NAND2_X1 ZN output drv=1 | OR 1.30435469e+09/0.683743894 X 109687232/0.981026351 ratio 0.08409 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_10
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g214/A2 | NAND2_X1 A2 input drv=0 | OR 475770560/0.944696367 X 5116521/0.999583781 ratio 0.01075 net=bp_processor/ic/node_1__io/io_resp_router/fifo_valid_lo[1]
choose input score=1.22534: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g214/A1
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g192/ZN | OR2_X1 ZN output drv=1 | OR 1.21211533e+09/0.334770113 X 109635744/0.018981576 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__data_sel_lo[1]
## depth 2: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g192 OR2_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g192/ZN | OR2_X1 ZN output drv=1 | OR 1.21211533e+09/0.334770113 X 109635744/0.018981576 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__data_sel_lo[1]
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g192/A1 | OR2_X1 A1 input drv=0 | OR 1.21023155e+09/0.334695518 X 109635720/0.018981576 ratio 0.09059 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/grants_lo[1]
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g192/A2 | OR2_X1 A2 input drv=0 | OR 3035457.25/0.000112116337 X 26.8712273/0 ratio 8.852e-06 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_1
choose input score=1.22512: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g192/A1
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g521/ZN | NOR2_X1 ZN output drv=1 | OR 1.21023155e+09/0.334695518 X 109635720/0.018981576 ratio 0.09059 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/grants_lo[1]
## depth 3: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g521 NOR2_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g521/A2 | NOR2_X1 A2 input drv=0 | OR 1.1766592e+09/0.588870287 X 114150464/0.978762031 ratio 0.09701 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_17
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g521/ZN | NOR2_X1 ZN output drv=1 | OR 1.21023155e+09/0.334695518 X 109635720/0.018981576 ratio 0.09059 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/grants_lo[1]
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g521/A1 | NOR2_X1 A1 input drv=0 | OR 613743232/0.185912609 X 358466752/0.106244445 ratio 0.5841 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_8
choose input score=1.29288: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g521/A2
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g524/ZN | NAND3_X1 ZN output drv=1 | OR 1.1766592e+09/0.588870287 X 114150464/0.978762031 ratio 0.09701 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_17
## depth 4: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g524 NAND3_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g524/ZN | NAND3_X1 ZN output drv=1 | OR 1.1766592e+09/0.588870287 X 114150464/0.978762031 ratio 0.09701 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_17
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g524/A2 | NAND3_X1 A2 input drv=0 | OR 701762816/0.580734968 X 290783040/0.113533974 ratio 0.4144 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_14
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g524/A3 | NAND3_X1 A3 input drv=0 | OR 484567776/0.944338083 X 5118496/0.999583721 ratio 0.01056 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_1__reqs_lo[0]
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g524/A1 | NAND3_X1 A1 input drv=0 | OR 854990272/0.749675691 X 525585600/0.187140822 ratio 0.6147 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_13
choose input score=1.05284: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g524/A2
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g527/ZN | OAI21_X1 ZN output drv=1 | OR 701762816/0.580734968 X 290783040/0.113533974 ratio 0.4144 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_14
## depth 5: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g527 OAI21_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g527/A | OAI21_X1 A input drv=0 | OR 695770944/0.419500589 X 286902784/0.88676095 ratio 0.4124 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g527/ZN | OAI21_X1 ZN output drv=1 | OR 701762816/0.580734968 X 290783040/0.113533974 ratio 0.4144 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_14
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g527/B2 | OAI21_X1 B2 input drv=0 | OR 19726998/0.999237418 X 5162626.5/0.999589741 ratio 0.2617 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_2__reqs_lo[0]
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g527/B1 | OAI21_X1 B1 input drv=0 | OR 900015552/0.263525158 X 726283840/0.189294368 ratio 0.807 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__reqs_lo[0]
choose input score=1.05491: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g527/A
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg_reg/Q | DFF_X1 Q output drv=1 | OR 695770944/0.419500589 X 286902784/0.88676095 ratio 0.4124 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg
## depth 6: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg_reg DFF_X1, current Q
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg_reg/D | DFF_X1 D input drv=0 | OR 1.82294938e+09/0.415680647 X 623264256/0.834528565 ratio 0.3419 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_5
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg_reg/QN | DFF_X1 QN output drv=1 | OR 695770944/0.580499411 X 286902784/0.11323905 ratio 0.4124 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_1
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg_reg/Q | DFF_X1 Q output drv=1 | OR 695770944/0.419500589 X 286902784/0.88676095 ratio 0.4124 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg_reg/CK | DFF_X1 CK input drv=0 | OR 1.42857139e+09/0.5 X 1.42857152e+09/0.5 ratio 1 net=p_clk_C_i
choose input score=1.07695: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/last_r_0_sv2v_reg_reg/D
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g490/ZN | NOR2_X1 ZN output drv=1 | OR 1.82294938e+09/0.415680647 X 623264256/0.834528565 ratio 0.3419 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_5
## depth 7: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g490 NOR2_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g490/A1 | NOR2_X1 A1 input drv=0 | OR 1.82268621e+09/0.584286571 X 622496448/0.165405616 ratio 0.3415 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_3
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g490/ZN | NOR2_X1 ZN output drv=1 | OR 1.82294938e+09/0.415680647 X 623264256/0.834528565 ratio 0.3419 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_5
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g490/A2 | NOR2_X1 A2 input drv=0 | OR 978748.312/7.8856945e-05 X 978777.938/7.8856945e-05 ratio 1 net=FE_OFN48254_n
choose input score=1.07735: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g490/A1
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g492/Z | MUX2_X1 Z output drv=1 | OR 1.82268621e+09/0.584286571 X 622496448/0.165405616 ratio 0.3415 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_3
## depth 8: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g492 MUX2_X1, current Z
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g492/B | MUX2_X1 B input drv=0 | OR 1.1766592e+09/0.588870287 X 114150464/0.978762031 ratio 0.09701 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_17
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g492/S | MUX2_X1 S input drv=0 | OR 1.87228006e+09/0.45242101 X 397555648/0.0602717325 ratio 0.2123 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/_0_net_
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g492/Z | MUX2_X1 Z output drv=1 | OR 1.82268621e+09/0.584286571 X 622496448/0.165405616 ratio 0.3415 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_3
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g492/A | MUX2_X1 A input drv=0 | OR 695770944/0.580499411 X 286902784/0.11323905 ratio 0.4124 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_1
choose input score=1.29288: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g492/B
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g524/ZN | NAND3_X1 ZN output drv=1 | OR 1.1766592e+09/0.588870287 X 114150464/0.978762031 ratio 0.09701 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/n_17
LOOP bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g524/ZN
