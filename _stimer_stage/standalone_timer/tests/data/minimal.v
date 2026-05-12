module minimal (IN, OUT);
input IN;
output OUT;
wire net0;
NAND2_X1 U1 (.A(IN), .B(IN), .Z(OUT));
endmodule
