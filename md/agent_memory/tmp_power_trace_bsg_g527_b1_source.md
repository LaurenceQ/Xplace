target bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/brr/g527/B1 | OAI21_X1 B1 input drv=0 | OR 900015552/0.263525158 X 726283840/0.189294368 ratio 0.807 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__reqs_lo[0]
driver candidate bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g125/ZN | AND2_X1 ZN output drv=1 | OR 900015552/0.263525158 X 726283840/0.189294368 ratio 0.807 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__reqs_lo[0]
## depth 0: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g125 AND2_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g125/ZN | AND2_X1 ZN output drv=1 | OR 900015552/0.263525158 X 726283840/0.189294368 ratio 0.807 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__reqs_lo[0]
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g125/A2 | AND2_X1 A2 input drv=0 | OR 900015552/0.263525158 X 726283840/0.189294368 ratio 0.807 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/n_1
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g125/A1 | AND2_X1 A1 input drv=0 | OR 0/1 X 0/1 ratio nan net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__decoded_dest_lo[0]
choose input score=0.267263: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g125/A2
driver: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g126/ZN | AND2_X1 ZN output drv=1 | OR 900015552/0.263525158 X 726283840/0.189294368 ratio 0.807 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/n_1
## depth 1: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g126 AND2_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g126/A2 | AND2_X1 A2 input drv=0 | OR 271377600/0.977459133 X 24489936/0.998884201 ratio 0.09024 net=bp_processor/ic/node_1__io/io_resp_router/fifo_valid_lo[0]
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g126/ZN | AND2_X1 ZN output drv=1 | OR 900015552/0.263525158 X 726283840/0.189294368 ratio 0.807 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/n_1
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g126/A1 | AND2_X1 A1 input drv=0 | OR 845919296/0.269602239 X 722448960/0.189505816 ratio 0.854 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/FE_OFN48231_releases_0
choose input score=0.931182: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__wic/g126/A2
driver: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/QN | DFF_X1 QN output drv=1 | OR 271377600/0.977459133 X 24489936/0.998884201 ratio 0.09024 net=bp_processor/ic/node_1__io/io_resp_router/fifo_valid_lo[0]
## depth 2: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg DFF_X1, current QN
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/D | DFF_X1 D input drv=0 | OR 321147808/0.0259206891 X 29047436/0.00131177902 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_7
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/QN | DFF_X1 QN output drv=1 | OR 271377600/0.977459133 X 24489936/0.998884201 ratio 0.09024 net=bp_processor/ic/node_1__io/io_resp_router/fifo_valid_lo[0]
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/Q | DFF_X1 Q output drv=1 | OR 271377600/0.0225408673 X 24489936/0.00111579895 ratio 0.09024 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/CK | DFF_X1 CK input drv=0 | OR 1.42857139e+09/0.5 X 1.42857152e+09/0.5 ratio 1 net=p_clk_C_i
choose input score=0.93416: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/D
driver: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g378/ZN | INV_X1 ZN output drv=1 | OR 321147808/0.0259206891 X 29047436/0.00131177902 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_7
## depth 3: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g378 INV_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g378/A | INV_X1 A input drv=0 | OR 321147808/0.974079311 X 29047436/0.998688221 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_5
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g378/ZN | INV_X1 ZN output drv=1 | OR 321147808/0.0259206891 X 29047436/0.00131177902 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_7
choose input score=0.93416: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g378/A
driver: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380/ZN | AOI221_X1 ZN output drv=1 | OR 321147808/0.974079311 X 29047436/0.998688221 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_5
## depth 4: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380 AOI221_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380/ZN | AOI221_X1 ZN output drv=1 | OR 321147808/0.974079311 X 29047436/0.998688221 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_5
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380/B2 | AOI221_X1 B2 input drv=0 | OR 271377600/0.0225408673 X 24489936/0.00111579895 ratio 0.09024 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380/C1 | AOI221_X1 C1 input drv=0 | OR 1.25524608e+09/0.248562157 X 381340352/0.0462913103 ratio 0.3038 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__any_yumi
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380/C2 | AOI221_X1 C2 input drv=0 | OR 248972848/0.0382995009 X 58802244/0.00352406502 ratio 0.2362 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_1
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380/B1 | AOI221_X1 B1 input drv=0 | OR 696055232/0.731158197 X 473454592/0.959013581 ratio 0.6802 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_13
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380/A | AOI221_X1 A input drv=0 | OR 978748.312/7.8856945e-05 X 978777.938/7.8856945e-05 ratio 1 net=bp_processor/FE_OFN2363_router_tag_data_lo_8
choose input score=0.931182: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380/B2
driver: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/Q | DFF_X1 Q output drv=1 | OR 271377600/0.0225408673 X 24489936/0.00111579895 ratio 0.09024 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg
## depth 5: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg DFF_X1, current Q
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/D | DFF_X1 D input drv=0 | OR 321147808/0.0259206891 X 29047436/0.00131177902 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_7
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/QN | DFF_X1 QN output drv=1 | OR 271377600/0.977459133 X 24489936/0.998884201 ratio 0.09024 net=bp_processor/ic/node_1__io/io_resp_router/fifo_valid_lo[0]
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/Q | DFF_X1 Q output drv=1 | OR 271377600/0.0225408673 X 24489936/0.00111579895 ratio 0.09024 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg
- bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/CK | DFF_X1 CK input drv=0 | OR 1.42857139e+09/0.5 X 1.42857152e+09/0.5 ratio 1 net=p_clk_C_i
choose input score=0.93416: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/empty_r_sv2v_reg_reg/D
driver: bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g378/ZN | INV_X1 ZN output drv=1 | OR 321147808/0.0259206891 X 29047436/0.00131177902 ratio 0.09045 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/n_7
LOOP bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g378/ZN
