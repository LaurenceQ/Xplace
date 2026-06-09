target bp_processor/ic/node_1__io/io_resp_router/in_ch_0__twofer/g380/C1 | AOI221_X1 C1 input drv=0 | OR 1.25524608e+09/0.248562157 X 381340352/0.0462913103 ratio 0.3038 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__any_yumi
driver candidate bp_processor/ic/node_1__io/io_resp_router/g16/ZN | OR3_X1 ZN output drv=1 | OR 1.25524608e+09/0.248562157 X 381340352/0.0462913103 ratio 0.3038 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__any_yumi
## depth 0: bp_processor/ic/node_1__io/io_resp_router/g16 OR3_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/g16/A2 | OR3_X1 A2 input drv=0 | OR 940730112/0.202965736 X 201527072/0.0308053493 ratio 0.2142 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__yumis_lo[0]
- bp_processor/ic/node_1__io/io_resp_router/g16/ZN | OR3_X1 ZN output drv=1 | OR 1.25524608e+09/0.248562157 X 381340352/0.0462913103 ratio 0.3038 net=bp_processor/ic/node_1__io/io_resp_router/in_ch_0__any_yumi
- bp_processor/ic/node_1__io/io_resp_router/g16/A3 | OR3_X1 A3 input drv=0 | OR 358423552/0.048026111 X 66847476/0.00321549154 ratio 0.1865 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_1__yumis_lo[0]
- bp_processor/ic/node_1__io/io_resp_router/g16/A1 | OR3_X1 A1 input drv=0 | OR 112568880/0.00964468718 X 123255664/0.0128038526 ratio 1.095 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_2__yumis_lo[0]
choose input score=0.957936: bp_processor/ic/node_1__io/io_resp_router/g16/A2
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g209/ZN | NOR2_X1 ZN output drv=1 | OR 940730112/0.202965736 X 201527072/0.0308053493 ratio 0.2142 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__yumis_lo[0]
## depth 1: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g209 NOR2_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g209/A2 | NOR2_X1 A2 input drv=0 | OR 451123648/0.0978844762 X 691822784/0.746331394 ratio 1.534 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_4
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g209/ZN | NOR2_X1 ZN output drv=1 | OR 940730112/0.202965736 X 201527072/0.0308053493 ratio 0.2142 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__yumis_lo[0]
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g209/A1 | NOR2_X1 A1 input drv=0 | OR 930293696/0.775011361 X 463252640/0.878560781 ratio 0.498 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_11
choose input score=1.182: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g209/A2
driver: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g210/ZN | INV_X1 ZN output drv=1 | OR 451123648/0.0978844762 X 691822784/0.746331394 ratio 1.534 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_4
## depth 2: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g210 INV_X1, current ZN
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g210/A | INV_X1 A input drv=0 | OR 451123648/0.902115524 X 691822784/0.253668606 ratio 1.534 net=bp_processor/ic/node_1__io/io_resp_link_li[62]
- bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g210/ZN | INV_X1 ZN output drv=1 | OR 451123648/0.0978844762 X 691822784/0.746331394 ratio 1.534 net=bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/n_4
choose input score=1.182: bp_processor/ic/node_1__io/io_resp_router/out_ch_0__woc/g210/A
driver: bp_processor/ic/node_1__io/io_async_io_resp_link/g17/ZN | INV_X1 ZN output drv=1 | OR 451123648/0.902115524 X 691822784/0.253668606 ratio 1.534 net=bp_processor/ic/node_1__io/io_resp_link_li[62]
## depth 3: bp_processor/ic/node_1__io/io_async_io_resp_link/g17 INV_X1, current ZN
- bp_processor/ic/node_1__io/io_async_io_resp_link/g17/ZN | INV_X1 ZN output drv=1 | OR 451123648/0.902115524 X 691822784/0.253668606 ratio 1.534 net=bp_processor/ic/node_1__io/io_resp_link_li[62]
- bp_processor/ic/node_1__io/io_async_io_resp_link/g17/A | INV_X1 A input drv=0 | OR 451123648/0.0978844762 X 691822784/0.746331394 ratio 1.534 net=bp_processor/ic/node_1__io/io_async_io_resp_link/blink_full_lo
choose input score=1.182: bp_processor/ic/node_1__io/io_async_io_resp_link/g17/A
driver: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g134/ZN | NOR4_X1 ZN output drv=1 | OR 451123648/0.0978844762 X 691822784/0.746331394 ratio 1.534 net=bp_processor/ic/node_1__io/io_async_io_resp_link/blink_full_lo
## depth 4: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g134 NOR4_X1, current ZN
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g134/ZN | NOR4_X1 ZN output drv=1 | OR 451123648/0.0978844762 X 691822784/0.746331394 ratio 1.534 net=bp_processor/ic/node_1__io/io_async_io_resp_link/blink_full_lo
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g134/A3 | NOR4_X1 A3 input drv=0 | OR 244646448/0.094569087 X 9034916/0.000239372253 ratio 0.03693 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/n_4
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g134/A2 | NOR4_X1 A2 input drv=0 | OR 608785024/0.307323098 X 157558608/0.00773751736 ratio 0.2588 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/n_5
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g134/A4 | NOR4_X1 A4 input drv=0 | OR 684567488/0.602958918 X 320546944/0.128506064 ratio 0.4682 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/n_6
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g134/A1 | NOR4_X1 A1 input drv=0 | OR 682198592/0.606909513 X 337819488/0.13673377 ratio 0.4952 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/n_7
choose input score=1.0574: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g134/A3
driver: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g138/ZN | XNOR2_X1 ZN output drv=1 | OR 244646448/0.094569087 X 9034916/0.000239372253 ratio 0.03693 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/n_4
## depth 5: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g138 XNOR2_X1, current ZN
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g138/A | XNOR2_X1 A input drv=0 | OR 244645072/0.094569087 X 9033539/0.000239372253 ratio 0.03693 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/w_ptr_gray_r[3]
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g138/ZN | XNOR2_X1 ZN output drv=1 | OR 244646448/0.094569087 X 9034916/0.000239372253 ratio 0.03693 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/n_4
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g138/B | XNOR2_X1 B input drv=0 | OR 1376.88452/1 X 1376.88318/1 ratio 1 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/r_ptr_gray_r_wsync[3]
choose input score=1.0574: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/g138/A
driver: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/iclk_data_o_3_sv2v_reg_reg/Q | DFF_X1 Q output drv=1 | OR 244645072/0.094569087 X 9033539/0.000239372253 ratio 0.03693 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/w_ptr_gray_r[3]
## depth 6: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/iclk_data_o_3_sv2v_reg_reg DFF_X1, current Q
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/iclk_data_o_3_sv2v_reg_reg/D | DFF_X1 D input drv=0 | OR 771716736/0.130746305 X 20660004/0.0006737113 ratio 0.02677 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/n_4
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/iclk_data_o_3_sv2v_reg_reg/QN | DFF_X1 QN output drv=1 | OR 244645072/0.905430913 X 9033539/0.999760628 ratio 0.03693 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/UNCONNECTED253583
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/iclk_data_o_3_sv2v_reg_reg/Q | DFF_X1 Q output drv=1 | OR 244645072/0.094569087 X 9033539/0.000239372253 ratio 0.03693 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/w_ptr_gray_r[3]
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/iclk_data_o_3_sv2v_reg_reg/CK | DFF_X1 CK input drv=0 | OR 1.42857139e+09/0.5 X 1.42857152e+09/0.5 ratio 1 net=p_clk_C_i
choose input score=1.1033: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/iclk_data_o_3_sv2v_reg_reg/D
driver: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g101/ZN | NOR2_X1 ZN output drv=1 | OR 771716736/0.130746305 X 20660004/0.0006737113 ratio 0.02677 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/n_4
## depth 7: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g101 NOR2_X1, current ZN
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g101/A1 | NOR2_X1 A1 input drv=0 | OR 771649600/0.869243383 X 20660974/0.999326229 ratio 0.02678 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/n_0
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g101/ZN | NOR2_X1 ZN output drv=1 | OR 771716736/0.130746305 X 20660004/0.0006737113 ratio 0.02677 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/n_4
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g101/A2 | NOR2_X1 A2 input drv=0 | OR 978748.312/7.8856945e-05 X 978777.938/7.8856945e-05 ratio 1 net=bp_processor/FE_OFN48319_router_tag_data_lo_8
choose input score=1.10331: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g101/A1
driver: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g105/ZN | INV_X1 ZN output drv=1 | OR 771649600/0.869243383 X 20660974/0.999326229 ratio 0.02678 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/n_0
## depth 8: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g105 INV_X1, current ZN
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g105/ZN | INV_X1 ZN output drv=1 | OR 771649600/0.869243383 X 20660974/0.999326229 ratio 0.02678 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/n_0
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g105/A | INV_X1 A input drv=0 | OR 771649600/0.130756617 X 20660974/0.000673770905 ratio 0.02678 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/w_ptr_gray_n[3]
choose input score=1.10331: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/ptr_sync/sync_p_z_blss/g105/A
driver: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/g588/ZN | OAI21_X1 ZN output drv=1 | OR 771649600/0.130756617 X 20660974/0.000673770905 ratio 0.02678 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/w_ptr_gray_n[3]
## depth 9: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/g588 OAI21_X1, current ZN
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/g588/B1 | OAI21_X1 B1 input drv=0 | OR 1.88087718e+09/0.555740118 X 417767392/0.0674364865 ratio 0.2221 net=bp_processor/ic/node_1__io/io_async_io_resp_link/blink_enq_li
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/g588/ZN | OAI21_X1 ZN output drv=1 | OR 771649600/0.130756617 X 20660974/0.000673770905 ratio 0.02678 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/w_ptr_gray_n[3]
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/g588/A | OAI21_X1 A input drv=0 | OR 534074432/0.907364726 X 12143180/0.99954927 ratio 0.02274 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/n_23
- bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/g588/B2 | OAI21_X1 B2 input drv=0 | OR 244645072/0.905430913 X 9033539/0.999760628 ratio 0.03693 net=bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/n_20
choose input score=1.26619: bp_processor/ic/node_1__io/io_async_io_resp_link/link_b_to_a/bapg_wr/g588/B1
driver: bp_processor/ic/node_1__io/io_async_io_resp_link/g21/ZN | AND2_X1 ZN output drv=1 | OR 1.88087718e+09/0.555740118 X 417767392/0.0674364865 ratio 0.2221 net=bp_processor/ic/node_1__io/io_async_io_resp_link/blink_enq_li
## depth 10: bp_processor/ic/node_1__io/io_async_io_resp_link/g21 AND2_X1, current ZN
- bp_processor/ic/node_1__io/io_async_io_resp_link/g21/ZN | AND2_X1 ZN output drv=1 | OR 1.88087718e+09/0.555740118 X 417767392/0.0674364865 ratio 0.2221 net=bp_processor/ic/node_1__io/io_async_io_resp_link/blink_enq_li
- bp_processor/ic/node_1__io/io_async_io_resp_link/g21/A2 | AND2_X1 A2 input drv=0 | OR 451123648/0.902115524 X 691822784/0.253668606 ratio 1.534 net=bp_processor/ic/node_1__io/io_resp_link_li[62]
- bp_processor/ic/node_1__io/io_async_io_resp_link/g21/A1 | AND2_X1 A1 input drv=0 | OR 1.77689715e+09/0.616040945 X 921871616/0.265844822 ratio 0.5188 net=bp_processor/ic/node_1__io/io_resp_link_lo[63]
choose input score=1.182: bp_processor/ic/node_1__io/io_async_io_resp_link/g21/A2
driver: bp_processor/ic/node_1__io/io_async_io_resp_link/g17/ZN | INV_X1 ZN output drv=1 | OR 451123648/0.902115524 X 691822784/0.253668606 ratio 1.534 net=bp_processor/ic/node_1__io/io_resp_link_li[62]
LOOP bp_processor/ic/node_1__io/io_async_io_resp_link/g17/ZN
