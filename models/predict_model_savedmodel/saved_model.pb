��	
�#�"
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( "
grad_xbool( "
grad_ybool( 
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028	
�
0bidirectional_12/backward_lstm_12/lstm_cell/biasVarHandleOp*
_output_shapes
: *A

debug_name31bidirectional_12/backward_lstm_12/lstm_cell/bias/*
dtype0*
shape:�*A
shared_name20bidirectional_12/backward_lstm_12/lstm_cell/bias
�
Dbidirectional_12/backward_lstm_12/lstm_cell/bias/Read/ReadVariableOpReadVariableOp0bidirectional_12/backward_lstm_12/lstm_cell/bias*
_output_shapes	
:�*
dtype0
�
;bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *L

debug_name><bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel/*
dtype0*
shape:
��*L
shared_name=;bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel
�
Obidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp;bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
dense_9/biasVarHandleOp*
_output_shapes
: *

debug_namedense_9/bias/*
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
�
dense_9/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_9/kernel/*
dtype0*
shape:	�*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	�*
dtype0
�
<bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *M

debug_name?=bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel/*
dtype0*
shape:
��*M
shared_name><bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel
�
Pbidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp<bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
2bidirectional_12/backward_lstm_12/lstm_cell/kernelVarHandleOp*
_output_shapes
: *C

debug_name53bidirectional_12/backward_lstm_12/lstm_cell/kernel/*
dtype0*
shape:	�*C
shared_name42bidirectional_12/backward_lstm_12/lstm_cell/kernel
�
Fbidirectional_12/backward_lstm_12/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp2bidirectional_12/backward_lstm_12/lstm_cell/kernel*
_output_shapes
:	�*
dtype0
�
/bidirectional_12/forward_lstm_12/lstm_cell/biasVarHandleOp*
_output_shapes
: *@

debug_name20bidirectional_12/forward_lstm_12/lstm_cell/bias/*
dtype0*
shape:�*@
shared_name1/bidirectional_12/forward_lstm_12/lstm_cell/bias
�
Cbidirectional_12/forward_lstm_12/lstm_cell/bias/Read/ReadVariableOpReadVariableOp/bidirectional_12/forward_lstm_12/lstm_cell/bias*
_output_shapes	
:�*
dtype0
�
1bidirectional_12/forward_lstm_12/lstm_cell/kernelVarHandleOp*
_output_shapes
: *B

debug_name42bidirectional_12/forward_lstm_12/lstm_cell/kernel/*
dtype0*
shape:	�*B
shared_name31bidirectional_12/forward_lstm_12/lstm_cell/kernel
�
Ebidirectional_12/forward_lstm_12/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp1bidirectional_12/forward_lstm_12/lstm_cell/kernel*
_output_shapes
:	�*
dtype0
�
dense_9/bias_1VarHandleOp*
_output_shapes
: *

debug_namedense_9/bias_1/*
dtype0*
shape:*
shared_namedense_9/bias_1
m
"dense_9/bias_1/Read/ReadVariableOpReadVariableOpdense_9/bias_1*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpdense_9/bias_1*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
dense_9/kernel_1VarHandleOp*
_output_shapes
: *!

debug_namedense_9/kernel_1/*
dtype0*
shape:	�*!
shared_namedense_9/kernel_1
v
$dense_9/kernel_1/Read/ReadVariableOpReadVariableOpdense_9/kernel_1*
_output_shapes
:	�*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpdense_9/kernel_1*
_class
loc:@Variable_1*
_output_shapes
:	�*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:	�*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	�*
dtype0
�
&seed_generator_42/seed_generator_stateVarHandleOp*
_output_shapes
: *7

debug_name)'seed_generator_42/seed_generator_state/*
dtype0	*
shape:*7
shared_name(&seed_generator_42/seed_generator_state
�
:seed_generator_42/seed_generator_state/Read/ReadVariableOpReadVariableOp&seed_generator_42/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_2/Initializer/ReadVariableOpReadVariableOp&seed_generator_42/seed_generator_state*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0	
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0	*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0	
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
�
2bidirectional_12/backward_lstm_12/lstm_cell/bias_1VarHandleOp*
_output_shapes
: *C

debug_name53bidirectional_12/backward_lstm_12/lstm_cell/bias_1/*
dtype0*
shape:�*C
shared_name42bidirectional_12/backward_lstm_12/lstm_cell/bias_1
�
Fbidirectional_12/backward_lstm_12/lstm_cell/bias_1/Read/ReadVariableOpReadVariableOp2bidirectional_12/backward_lstm_12/lstm_cell/bias_1*
_output_shapes	
:�*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOp2bidirectional_12/backward_lstm_12/lstm_cell/bias_1*
_class
loc:@Variable_3*
_output_shapes	
:�*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:�*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
f
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes	
:�*
dtype0
�
>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1VarHandleOp*
_output_shapes
: *O

debug_nameA?bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1/*
dtype0*
shape:
��*O
shared_name@>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1
�
Rbidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1/Read/ReadVariableOpReadVariableOp>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1* 
_output_shapes
:
��*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOp>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1*
_class
loc:@Variable_4* 
_output_shapes
:
��*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:
��*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
k
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4* 
_output_shapes
:
��*
dtype0
�
4bidirectional_12/backward_lstm_12/lstm_cell/kernel_1VarHandleOp*
_output_shapes
: *E

debug_name75bidirectional_12/backward_lstm_12/lstm_cell/kernel_1/*
dtype0*
shape:	�*E
shared_name64bidirectional_12/backward_lstm_12/lstm_cell/kernel_1
�
Hbidirectional_12/backward_lstm_12/lstm_cell/kernel_1/Read/ReadVariableOpReadVariableOp4bidirectional_12/backward_lstm_12/lstm_cell/kernel_1*
_output_shapes
:	�*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOp4bidirectional_12/backward_lstm_12/lstm_cell/kernel_1*
_class
loc:@Variable_5*
_output_shapes
:	�*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:	�*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
j
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
:	�*
dtype0
�
&seed_generator_41/seed_generator_stateVarHandleOp*
_output_shapes
: *7

debug_name)'seed_generator_41/seed_generator_state/*
dtype0	*
shape:*7
shared_name(&seed_generator_41/seed_generator_state
�
:seed_generator_41/seed_generator_state/Read/ReadVariableOpReadVariableOp&seed_generator_41/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_6/Initializer/ReadVariableOpReadVariableOp&seed_generator_41/seed_generator_state*
_class
loc:@Variable_6*
_output_shapes
:*
dtype0	
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0	*
shape:*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0	
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:*
dtype0	
�
1bidirectional_12/forward_lstm_12/lstm_cell/bias_1VarHandleOp*
_output_shapes
: *B

debug_name42bidirectional_12/forward_lstm_12/lstm_cell/bias_1/*
dtype0*
shape:�*B
shared_name31bidirectional_12/forward_lstm_12/lstm_cell/bias_1
�
Ebidirectional_12/forward_lstm_12/lstm_cell/bias_1/Read/ReadVariableOpReadVariableOp1bidirectional_12/forward_lstm_12/lstm_cell/bias_1*
_output_shapes	
:�*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOp1bidirectional_12/forward_lstm_12/lstm_cell/bias_1*
_class
loc:@Variable_7*
_output_shapes	
:�*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:�*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
f
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes	
:�*
dtype0
�
=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_1VarHandleOp*
_output_shapes
: *N

debug_name@>bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_1/*
dtype0*
shape:
��*N
shared_name?=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_1
�
Qbidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_1/Read/ReadVariableOpReadVariableOp=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_1* 
_output_shapes
:
��*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOp=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_1*
_class
loc:@Variable_8* 
_output_shapes
:
��*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:
��*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
k
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8* 
_output_shapes
:
��*
dtype0
�
3bidirectional_12/forward_lstm_12/lstm_cell/kernel_1VarHandleOp*
_output_shapes
: *D

debug_name64bidirectional_12/forward_lstm_12/lstm_cell/kernel_1/*
dtype0*
shape:	�*D
shared_name53bidirectional_12/forward_lstm_12/lstm_cell/kernel_1
�
Gbidirectional_12/forward_lstm_12/lstm_cell/kernel_1/Read/ReadVariableOpReadVariableOp3bidirectional_12/forward_lstm_12/lstm_cell/kernel_1*
_output_shapes
:	�*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOp3bidirectional_12/forward_lstm_12/lstm_cell/kernel_1*
_class
loc:@Variable_9*
_output_shapes
:	�*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape:	�*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
j
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
:	�*
dtype0
}
serve_input_dataPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserve_input_data3bidirectional_12/forward_lstm_12/lstm_cell/kernel_1=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_11bidirectional_12/forward_lstm_12/lstm_cell/bias_14bidirectional_12/backward_lstm_12/lstm_cell/kernel_1>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_12bidirectional_12/backward_lstm_12/lstm_cell/bias_1dense_9/kernel_1dense_9/bias_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *5
f0R.
,__inference_signature_wrapper___call___85099
�
serving_default_input_dataPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_data3bidirectional_12/forward_lstm_12/lstm_cell/kernel_1=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_11bidirectional_12/forward_lstm_12/lstm_cell/bias_14bidirectional_12/backward_lstm_12/lstm_cell/kernel_1>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_12bidirectional_12/backward_lstm_12/lstm_cell/bias_1dense_9/kernel_1dense_9/bias_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *5
f0R.
,__inference_signature_wrapper___call___85120

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
J
0
	1

2
3
4
5
6
7
8
9*
<
0
	1

2
3
4
5
6
7*

0
1*
<
0
1
2
3
4
5
6
7*
* 

trace_0* 
"
	serve
serving_default* 
JD
VARIABLE_VALUE
Variable_9&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_8&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_7&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_6&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_5&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_4&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_3&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_2&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_1&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEVariable&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE3bidirectional_12/forward_lstm_12/lstm_cell/kernel_1+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE1bidirectional_12/forward_lstm_12/lstm_cell/bias_1+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE4bidirectional_12/backward_lstm_12/lstm_cell/kernel_1+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUE>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_9/kernel_1+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_9/bias_1+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUE=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_1+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE2bidirectional_12/backward_lstm_12/lstm_cell/bias_1+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable3bidirectional_12/forward_lstm_12/lstm_cell/kernel_11bidirectional_12/forward_lstm_12/lstm_cell/bias_14bidirectional_12/backward_lstm_12/lstm_cell/kernel_1>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1dense_9/kernel_1dense_9/bias_1=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_12bidirectional_12/backward_lstm_12/lstm_cell/bias_1Const*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *'
f"R 
__inference__traced_save_85292
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable3bidirectional_12/forward_lstm_12/lstm_cell/kernel_11bidirectional_12/forward_lstm_12/lstm_cell/bias_14bidirectional_12/backward_lstm_12/lstm_cell/kernel_1>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1dense_9/kernel_1dense_9/bias_1=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_12bidirectional_12/backward_lstm_12/lstm_cell/bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J **
f%R#
!__inference__traced_restore_85355��
��
�
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_body_84783�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_loop_counterw
sfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_maxJ
Ffunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholderL
Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_1L
Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_2L
Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_3L
Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_4�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor_0�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensor_0x
efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resource_0:	�{
gfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resource_0:
��u
ffunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resource_0:	��
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_forward_lstm_12_1_zeros_like_0G
Cfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identityI
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_1I
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_2I
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_3I
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_4I
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_5I
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_6�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensorv
cfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resource:	�y
efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resource:
��s
dfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resource:	��
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_forward_lstm_12_1_zeros_like��Zfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOp�\functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp�[functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp�
lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor_0Ffunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholderufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
`functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensor_0Ffunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholderwfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
Zfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOpefunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
Mfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/MatMulMatMulefunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read/TensorListGetItem:item:0bfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
\functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpgfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
Ofunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/MatMul_1MatMulHfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_3dfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/addAddV2Wfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/MatMul:product:0Yfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
[functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOpffunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_1AddV2Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add:z:0cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Vfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/splitSplit_functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/split/split_dim:output:0Pfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/SigmoidSigmoidUfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
Pfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Sigmoid_1SigmoidUfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/mulMulTfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Sigmoid_1:y:0Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_4*
T0*(
_output_shapes
:�����������
Kfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/TanhTanhUfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/mul_1MulRfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Sigmoid:y:0Ofunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_2AddV2Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/mul:z:0Pfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
Pfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Sigmoid_2SigmoidUfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
Mfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Tanh_1TanhPfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/mul_2MulTfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Sigmoid_2:y:0Qfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
Ifunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
?functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TileTilegfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Rfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/SelectV2SelectV2Hfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile:output:0Pfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/mul_2:z:0�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_forward_lstm_12_1_zeros_like_0*
T0*(
_output_shapes
:�����������
Kfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
Afunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile_1Tilegfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Tfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
Kfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
Afunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile_2Tilegfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Tfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:����������
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/SelectV2_1SelectV2Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile_1:output:0Pfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/mul_2:z:0Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_3*
T0*(
_output_shapes
:�����������
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/SelectV2_2SelectV2Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Tile_2:output:0Pfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_2:z:0Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_4*
T0*(
_output_shapes
:�����������
_functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemHfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_1Ffunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholderLfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:����
@functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
>functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/addAddV2Ffunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholderIfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/add/y:output:0*
T0*
_output_shapes
: �
Bfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
@functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/add_1AddV2�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_loop_counterKfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/add_1/y:output:0*
T0*
_output_shapes
: �
Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/IdentityIdentityDfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/add_1:z:0@^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/NoOp*
T0*
_output_shapes
: �
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_1Identitysfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_max@^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/NoOp*
T0*
_output_shapes
: �
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_2IdentityBfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/add:z:0@^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/NoOp*
T0*
_output_shapes
: �
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_3Identityofunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0@^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/NoOp*
T0*
_output_shapes
: �
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_4IdentityLfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/SelectV2:output:0@^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/NoOp*
T0*(
_output_shapes
:�����������
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_5IdentityNfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/SelectV2_1:output:0@^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/NoOp*
T0*(
_output_shapes
:�����������
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_6IdentityNfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/SelectV2_2:output:0@^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/NoOp*
T0*(
_output_shapes
:�����������
?functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/NoOpNoOp[^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOp]^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp\^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 "�
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_1Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_1:output:0"�
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_2Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_2:output:0"�
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_3Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_3:output:0"�
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_4Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_4:output:0"�
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_5Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_5:output:0"�
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity_6Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity_6:output:0"�
Cfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identityLfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity:output:0"�
dfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resourceffunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resourcegfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
cfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resourceefunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resource_0"�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_forward_lstm_12_1_zeros_like�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_forward_lstm_12_1_zeros_like_0"�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensor�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensor_0"�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_forward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b: : : : :����������:����������:����������: : : : : :����������2�
Zfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOpZfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOp2�
\functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp\functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp2�
[functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp[functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp:��
(
_output_shapes
:����������
Y
_user_specified_nameA?functional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_like:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:��

_output_shapes
: 
x
_user_specified_name`^functional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensor:��

_output_shapes
: 
v
_user_specified_name^\functional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack/TensorListFromTensor:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: :pl

_output_shapes
: 
R
_user_specified_name:8functional_27_1/bidirectional_12_1/forward_lstm_12_1/Max: {

_output_shapes
: 
a
_user_specified_nameIGfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/loop_counter
�
�
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_cond_84782�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_loop_counterw
sfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_maxJ
Ffunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholderL
Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_1L
Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_2L
Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_3L
Hfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholder_4�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_cond_84782___redundant_placeholder0�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_cond_84782___redundant_placeholder1�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_cond_84782___redundant_placeholder2�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_cond_84782___redundant_placeholder3�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_cond_84782___redundant_placeholder4�
�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_cond_84782___redundant_placeholder5G
Cfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identity
�
Afunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Less/yConst*
_output_shapes
: *
dtype0*
value
B :��
?functional_27_1/bidirectional_12_1/forward_lstm_12_1/while/LessLessFfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_placeholderJfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Less/y:output:0*
T0*
_output_shapes
: �
Afunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Less_1Less�functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_while_loop_countersfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_functional_27_1_bidirectional_12_1_forward_lstm_12_1_max*
T0*
_output_shapes
: �
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/LogicalAnd
LogicalAndEfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Less_1:z:0Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Less:z:0*
_output_shapes
: �
Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/IdentityIdentityIfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "�
Cfunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_identityLfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\: : : : :����������:����������:����������:::::::

_output_shapes
::

_output_shapes
::

_output_shapes
::.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: :pl

_output_shapes
: 
R
_user_specified_name:8functional_27_1/bidirectional_12_1/forward_lstm_12_1/Max: {

_output_shapes
: 
a
_user_specified_nameIGfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/loop_counter
��
�
__inference__traced_save_85292
file_prefix4
!read_disablecopyonread_variable_9:	�7
#read_1_disablecopyonread_variable_8:
��2
#read_2_disablecopyonread_variable_7:	�1
#read_3_disablecopyonread_variable_6:	6
#read_4_disablecopyonread_variable_5:	�7
#read_5_disablecopyonread_variable_4:
��2
#read_6_disablecopyonread_variable_3:	�1
#read_7_disablecopyonread_variable_2:	6
#read_8_disablecopyonread_variable_1:	�/
!read_9_disablecopyonread_variable:`
Mread_10_disablecopyonread_bidirectional_12_forward_lstm_12_lstm_cell_kernel_1:	�Z
Kread_11_disablecopyonread_bidirectional_12_forward_lstm_12_lstm_cell_bias_1:	�a
Nread_12_disablecopyonread_bidirectional_12_backward_lstm_12_lstm_cell_kernel_1:	�l
Xread_13_disablecopyonread_bidirectional_12_backward_lstm_12_lstm_cell_recurrent_kernel_1:
��=
*read_14_disablecopyonread_dense_9_kernel_1:	�6
(read_15_disablecopyonread_dense_9_bias_1:k
Wread_16_disablecopyonread_bidirectional_12_forward_lstm_12_lstm_cell_recurrent_kernel_1:
��[
Lread_17_disablecopyonread_bidirectional_12_backward_lstm_12_lstm_cell_bias_1:	�
savev2_const
identity_37��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_9*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_9^Read/DisableCopyOnRead*
_output_shapes
:	�*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_variable_8*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_variable_8^Read_1/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0`

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��e

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��h
Read_2/DisableCopyOnReadDisableCopyOnRead#read_2_disablecopyonread_variable_7*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp#read_2_disablecopyonread_variable_7^Read_2/DisableCopyOnRead*
_output_shapes	
:�*
dtype0[

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:�`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:�h
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_variable_6*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_variable_6^Read_3/DisableCopyOnRead*
_output_shapes
:*
dtype0	Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0	*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0	*
_output_shapes
:h
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_variable_5*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_variable_5^Read_4/DisableCopyOnRead*
_output_shapes
:	�*
dtype0_

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_variable_4*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_variable_4^Read_5/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��h
Read_6/DisableCopyOnReadDisableCopyOnRead#read_6_disablecopyonread_variable_3*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp#read_6_disablecopyonread_variable_3^Read_6/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:�h
Read_7/DisableCopyOnReadDisableCopyOnRead#read_7_disablecopyonread_variable_2*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp#read_7_disablecopyonread_variable_2^Read_7/DisableCopyOnRead*
_output_shapes
:*
dtype0	[
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0	*
_output_shapes
:h
Read_8/DisableCopyOnReadDisableCopyOnRead#read_8_disablecopyonread_variable_1*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp#read_8_disablecopyonread_variable_1^Read_8/DisableCopyOnRead*
_output_shapes
:	�*
dtype0`
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Read_9/DisableCopyOnReadDisableCopyOnRead!read_9_disablecopyonread_variable*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp!read_9_disablecopyonread_variable^Read_9/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnReadMread_10_disablecopyonread_bidirectional_12_forward_lstm_12_lstm_cell_kernel_1*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpMread_10_disablecopyonread_bidirectional_12_forward_lstm_12_lstm_cell_kernel_1^Read_10/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_11/DisableCopyOnReadDisableCopyOnReadKread_11_disablecopyonread_bidirectional_12_forward_lstm_12_lstm_cell_bias_1*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpKread_11_disablecopyonread_bidirectional_12_forward_lstm_12_lstm_cell_bias_1^Read_11/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_12/DisableCopyOnReadDisableCopyOnReadNread_12_disablecopyonread_bidirectional_12_backward_lstm_12_lstm_cell_kernel_1*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpNread_12_disablecopyonread_bidirectional_12_backward_lstm_12_lstm_cell_kernel_1^Read_12/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_13/DisableCopyOnReadDisableCopyOnReadXread_13_disablecopyonread_bidirectional_12_backward_lstm_12_lstm_cell_recurrent_kernel_1*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpXread_13_disablecopyonread_bidirectional_12_backward_lstm_12_lstm_cell_recurrent_kernel_1^Read_13/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��p
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_dense_9_kernel_1*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_dense_9_kernel_1^Read_14/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	�n
Read_15/DisableCopyOnReadDisableCopyOnRead(read_15_disablecopyonread_dense_9_bias_1*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp(read_15_disablecopyonread_dense_9_bias_1^Read_15/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnReadWread_16_disablecopyonread_bidirectional_12_forward_lstm_12_lstm_cell_recurrent_kernel_1*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpWread_16_disablecopyonread_bidirectional_12_forward_lstm_12_lstm_cell_recurrent_kernel_1^Read_16/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_17/DisableCopyOnReadDisableCopyOnReadLread_17_disablecopyonread_bidirectional_12_backward_lstm_12_lstm_cell_bias_1*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpLread_17_disablecopyonread_bidirectional_12_backward_lstm_12_lstm_cell_bias_1^Read_17/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:�L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_36Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_37IdentityIdentity_36:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_37Identity_37:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:RN
L
_user_specified_name42bidirectional_12/backward_lstm_12/lstm_cell/bias_1:]Y
W
_user_specified_name?=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_1:.*
(
_user_specified_namedense_9/bias_1:0,
*
_user_specified_namedense_9/kernel_1:^Z
X
_user_specified_name@>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1:TP
N
_user_specified_name64bidirectional_12/backward_lstm_12/lstm_cell/kernel_1:QM
K
_user_specified_name31bidirectional_12/forward_lstm_12/lstm_cell/bias_1:SO
M
_user_specified_name53bidirectional_12/forward_lstm_12/lstm_cell/kernel_1:(
$
"
_user_specified_name
Variable:*	&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_cond_84962�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_loop_countery
ufunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_maxK
Gfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholderM
Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_1M
Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_2M
Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_3M
Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_4�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_cond_84962___redundant_placeholder0�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_cond_84962___redundant_placeholder1�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_cond_84962___redundant_placeholder2�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_cond_84962___redundant_placeholder3�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_cond_84962___redundant_placeholder4�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_cond_84962___redundant_placeholder5H
Dfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity
�
Bfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Less/yConst*
_output_shapes
: *
dtype0*
value
B :��
@functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/LessLessGfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholderKfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Less/y:output:0*
T0*
_output_shapes
: �
Bfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Less_1Less�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_loop_counterufunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_max*
T0*
_output_shapes
: �
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/LogicalAnd
LogicalAndFfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Less_1:z:0Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Less:z:0*
_output_shapes
: �
Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/IdentityIdentityJfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "�
Dfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identityMfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\: : : : :����������:����������:����������:::::::

_output_shapes
::

_output_shapes
::

_output_shapes
::.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: :qm

_output_shapes
: 
S
_user_specified_name;9functional_27_1/bidirectional_12_1/backward_lstm_12_1/Max:� |

_output_shapes
: 
b
_user_specified_nameJHfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/loop_counter
�
�
,__inference_signature_wrapper___call___85120

input_data
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *#
fR
__inference___call___85077t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85116:%!

_user_specified_name85114:%!

_user_specified_name85112:%!

_user_specified_name85110:%!

_user_specified_name85108:%!

_user_specified_name85106:%!

_user_specified_name85104:%!

_user_specified_name85102:X T
,
_output_shapes
:����������
$
_user_specified_name
input_data
��
�
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_body_84963�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_loop_countery
ufunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_maxK
Gfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholderM
Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_1M
Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_2M
Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_3M
Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_4�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor_0�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensor_0y
ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resource_0:	�|
hfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resource_0:
��v
gfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resource_0:	��
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_backward_lstm_12_1_zeros_like_0H
Dfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identityJ
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_1J
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_2J
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_3J
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_4J
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_5J
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_6�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensorw
dfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resource:	�z
ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resource:
��t
efunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resource:	��
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_backward_lstm_12_1_zeros_like��[functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOp�]functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp�\functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp�
mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
_functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor_0Gfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholdervfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
afunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensor_0Gfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholderxfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
[functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOpffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
Nfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/MatMulMatMulffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read/TensorListGetItem:item:0cfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
]functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOphfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
Pfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/MatMul_1MatMulIfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_3efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/addAddV2Xfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/MatMul:product:0Zfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
\functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOpgfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_1AddV2Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add:z:0dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Wfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/splitSplit`functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/split/split_dim:output:0Qfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/SigmoidSigmoidVfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
Qfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Sigmoid_1SigmoidVfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/mulMulUfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Sigmoid_1:y:0Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_4*
T0*(
_output_shapes
:�����������
Lfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/TanhTanhVfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/mul_1MulSfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Sigmoid:y:0Pfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_2AddV2Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/mul:z:0Qfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
Qfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Sigmoid_2SigmoidVfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
Nfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Tanh_1TanhQfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/mul_2MulUfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Sigmoid_2:y:0Rfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
Jfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
@functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TileTilehfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Sfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/SelectV2SelectV2Ifunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile:output:0Qfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/mul_2:z:0�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_backward_lstm_12_1_zeros_like_0*
T0*(
_output_shapes
:�����������
Lfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
Bfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile_1Tilehfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Ufunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
Lfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
Bfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile_2Tilehfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Read_1/TensorListGetItem:item:0Ufunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:����������
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/SelectV2_1SelectV2Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile_1:output:0Qfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/mul_2:z:0Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_3*
T0*(
_output_shapes
:�����������
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/SelectV2_2SelectV2Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Tile_2:output:0Qfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_2:z:0Ifunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_4*
T0*(
_output_shapes
:�����������
`functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemIfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholder_1Gfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholderMfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:����
Afunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
?functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/addAddV2Gfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_placeholderJfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/add/y:output:0*
T0*
_output_shapes
: �
Cfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
Afunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/add_1AddV2�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_loop_counterLfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/add_1/y:output:0*
T0*
_output_shapes
: �
Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/IdentityIdentityEfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/add_1:z:0A^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/NoOp*
T0*
_output_shapes
: �
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_1Identityufunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_functional_27_1_bidirectional_12_1_backward_lstm_12_1_maxA^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/NoOp*
T0*
_output_shapes
: �
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_2IdentityCfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/add:z:0A^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/NoOp*
T0*
_output_shapes
: �
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_3Identitypfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0A^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/NoOp*
T0*
_output_shapes
: �
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_4IdentityMfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/SelectV2:output:0A^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/NoOp*
T0*(
_output_shapes
:�����������
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_5IdentityOfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/SelectV2_1:output:0A^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/NoOp*
T0*(
_output_shapes
:�����������
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_6IdentityOfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/SelectV2_2:output:0A^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/NoOp*
T0*(
_output_shapes
:�����������
@functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/NoOpNoOp\^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOp^^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp]^functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 "�
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_1Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_1:output:0"�
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_2Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_2:output:0"�
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_3Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_3:output:0"�
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_4Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_4:output:0"�
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_5Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_5:output:0"�
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identity_6Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity_6:output:0"�
Dfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_identityMfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/Identity:output:0"�
efunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resourcegfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resourcehfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
dfunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resourceffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_lstm_cell_1_cast_readvariableop_resource_0"�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_backward_lstm_12_1_zeros_like�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_selectv2_functional_27_1_bidirectional_12_1_backward_lstm_12_1_zeros_like_0"�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensor�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_1_tensorlistfromtensor_0"�
�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor�functional_27_1_bidirectional_12_1_backward_lstm_12_1_while_tensorarrayv2read_tensorlistgetitem_functional_27_1_bidirectional_12_1_backward_lstm_12_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b: : : : :����������:����������:����������: : : : : :����������2�
[functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOp[functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast/ReadVariableOp2�
]functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp]functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/Cast_1/ReadVariableOp2�
\functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp\functional_27_1/bidirectional_12_1/backward_lstm_12_1/while/lstm_cell_1/add_1/ReadVariableOp:��
(
_output_shapes
:����������
Z
_user_specified_nameB@functional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_like:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:��

_output_shapes
: 
y
_user_specified_namea_functional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensor:��

_output_shapes
: 
w
_user_specified_name_]functional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack/TensorListFromTensor:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: :qm

_output_shapes
: 
S
_user_specified_name;9functional_27_1/bidirectional_12_1/backward_lstm_12_1/Max:� |

_output_shapes
: 
b
_user_specified_nameJHfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/loop_counter
�
�
,__inference_signature_wrapper___call___85099

input_data
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_dataunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *#
fR
__inference___call___85077t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name85095:%!

_user_specified_name85093:%!

_user_specified_name85091:%!

_user_specified_name85089:%!

_user_specified_name85087:%!

_user_specified_name85085:%!

_user_specified_name85083:%!

_user_specified_name85081:X T
,
_output_shapes
:����������
$
_user_specified_name
input_data
��
�
__inference___call___85077

input_datap
]functional_27_1_bidirectional_12_1_forward_lstm_12_1_lstm_cell_1_cast_readvariableop_resource:	�s
_functional_27_1_bidirectional_12_1_forward_lstm_12_1_lstm_cell_1_cast_1_readvariableop_resource:
��m
^functional_27_1_bidirectional_12_1_forward_lstm_12_1_lstm_cell_1_add_1_readvariableop_resource:	�q
^functional_27_1_bidirectional_12_1_backward_lstm_12_1_lstm_cell_1_cast_readvariableop_resource:	�t
`functional_27_1_bidirectional_12_1_backward_lstm_12_1_lstm_cell_1_cast_1_readvariableop_resource:
��n
_functional_27_1_bidirectional_12_1_backward_lstm_12_1_lstm_cell_1_add_1_readvariableop_resource:	�I
6functional_27_1_dense_9_1_cast_readvariableop_resource:	�G
9functional_27_1_dense_9_1_biasadd_readvariableop_resource:
identity��Ufunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOp�Wfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOp�Vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOp�;functional_27_1/bidirectional_12_1/backward_lstm_12_1/while�Tfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOp�Vfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOp�Ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOp�:functional_27_1/bidirectional_12_1/forward_lstm_12_1/while�0functional_27_1/dense_9_1/BiasAdd/ReadVariableOp�-functional_27_1/dense_9_1/Cast/ReadVariableOpZ
functional_27_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
functional_27_1/NotEqualNotEqual
input_datafunctional_27_1/Const:output:0*
T0*,
_output_shapes
:����������f
!functional_27_1/masking_8_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
$functional_27_1/masking_8_1/NotEqualNotEqual
input_data*functional_27_1/masking_8_1/Const:output:0*
T0*,
_output_shapes
:����������|
1functional_27_1/masking_8_1/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
functional_27_1/masking_8_1/AnyAny(functional_27_1/masking_8_1/NotEqual:z:0:functional_27_1/masking_8_1/Any/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(�
 functional_27_1/masking_8_1/CastCast(functional_27_1/masking_8_1/Any:output:0*

DstT0*

SrcT0
*,
_output_shapes
:�����������
functional_27_1/masking_8_1/mulMul
input_data$functional_27_1/masking_8_1/Cast:y:0*
T0*,
_output_shapes
:�����������
#functional_27_1/masking_8_1/SqueezeSqueeze(functional_27_1/masking_8_1/Any:output:0*
T0
*(
_output_shapes
:����������*
squeeze_dims
x
%functional_27_1/Any/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
functional_27_1/AnyAnyfunctional_27_1/NotEqual:z:0.functional_27_1/Any/reduction_indices:output:0*(
_output_shapes
:�����������
:functional_27_1/bidirectional_12_1/forward_lstm_12_1/ShapeShape#functional_27_1/masking_8_1/mul:z:0*
T0*
_output_shapes
::���
Hfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Bfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_sliceStridedSliceCfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/Shape:output:0Qfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice/stack:output:0Sfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice/stack_1:output:0Sfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
Afunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros/packedPackKfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice:output:0Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:�
@functional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
:functional_27_1/bidirectional_12_1/forward_lstm_12_1/zerosFillJfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros/packed:output:0Ifunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros/Const:output:0*
T0*(
_output_shapes
:�����������
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_1/packedPackKfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice:output:0Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:�
Bfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
<functional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_1FillLfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_1/packed:output:0Kfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:�����������
Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
Dfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_1StridedSlice#functional_27_1/masking_8_1/mul:z:0Sfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_1/stack:output:0Ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_1/stack_1:output:0Ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
>functional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose	Transpose#functional_27_1/masking_8_1/mul:z:0Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose/perm:output:0*
T0*,
_output_shapes
:�����������
Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
?functional_27_1/bidirectional_12_1/forward_lstm_12_1/ExpandDims
ExpandDimsfunctional_27_1/Any:output:0Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:�����������
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
@functional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose_1	TransposeHfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/ExpandDims:output:0Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose_1/perm:output:0*
T0
*,
_output_shapes
:�����������
Pfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Ofunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
Bfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2TensorListReserveYfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2/element_shape:output:0Xfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
\functional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorBfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose:y:0sfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Dfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_2StridedSliceBfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose:y:0Sfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_2/stack:output:0Ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_2/stack_1:output:0Ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
Tfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOpReadVariableOp]functional_27_1_bidirectional_12_1_forward_lstm_12_1_lstm_cell_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Gfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/MatMulMatMulMfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_2:output:0\functional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Vfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOp_functional_27_1_bidirectional_12_1_forward_lstm_12_1_lstm_cell_1_cast_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Ifunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/MatMul_1MatMulCfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros:output:0^functional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Dfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/addAddV2Qfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/MatMul:product:0Sfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOpReadVariableOp^functional_27_1_bidirectional_12_1_forward_lstm_12_1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Ffunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_1AddV2Hfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add:z:0]functional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Pfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Ffunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/splitSplitYfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/split/split_dim:output:0Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
Hfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/SigmoidSigmoidOfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Sigmoid_1SigmoidOfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
Dfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/mulMulNfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Sigmoid_1:y:0Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_1:output:0*
T0*(
_output_shapes
:�����������
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/TanhTanhOfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
Ffunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/mul_1MulLfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Sigmoid:y:0Ifunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
Ffunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_2AddV2Hfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/mul:z:0Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Sigmoid_2SigmoidOfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
Gfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Tanh_1TanhJfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
Ffunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/mul_2MulNfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Sigmoid_2:y:0Kfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
Rfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Qfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
Dfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_1TensorListReserve[functional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_1/element_shape:output:0Zfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���{
9functional_27_1/bidirectional_12_1/forward_lstm_12_1/timeConst*
_output_shapes
: *
dtype0*
value	B : �
?functional_27_1/bidirectional_12_1/forward_lstm_12_1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value
B :�{
9functional_27_1/bidirectional_12_1/forward_lstm_12_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
@functional_27_1/bidirectional_12_1/forward_lstm_12_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
@functional_27_1/bidirectional_12_1/forward_lstm_12_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
:functional_27_1/bidirectional_12_1/forward_lstm_12_1/rangeRangeIfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/range/start:output:0Bfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/Rank:output:0Ifunctional_27_1/bidirectional_12_1/forward_lstm_12_1/range/delta:output:0*
_output_shapes
: �
>functional_27_1/bidirectional_12_1/forward_lstm_12_1/Max/inputConst*
_output_shapes
: *
dtype0*
value
B :��
8functional_27_1/bidirectional_12_1/forward_lstm_12_1/MaxMaxGfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/Max/input:output:0Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/range:output:0*
T0*
_output_shapes
: �
Rfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Qfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
Dfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_2TensorListReserve[functional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_2/element_shape:output:0Zfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
^functional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorDfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose_1:y:0ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
?functional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_like	ZerosLikeJfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:�����������
Gfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
:functional_27_1/bidirectional_12_1/forward_lstm_12_1/whileWhilePfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while/loop_counter:output:0Afunctional_27_1/bidirectional_12_1/forward_lstm_12_1/Max:output:0Bfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/time:output:0Mfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2_1:handle:0Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_like:y:0Cfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros:output:0Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_1:output:0lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0]functional_27_1_bidirectional_12_1_forward_lstm_12_1_lstm_cell_1_cast_readvariableop_resource_functional_27_1_bidirectional_12_1_forward_lstm_12_1_lstm_cell_1_cast_1_readvariableop_resource^functional_27_1_bidirectional_12_1_forward_lstm_12_1_lstm_cell_1_add_1_readvariableop_resourceCfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/zeros_like:y:0*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*v
_output_shapesd
b: : : : :����������:����������:����������: : : : : :����������*%
_read_only_resource_inputs
	
*Q
bodyIRG
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_body_84783*Q
condIRG
Efunctional_27_1_bidirectional_12_1_forward_lstm_12_1_while_cond_84782*u
output_shapesd
b: : : : :����������:����������:����������: : : : : :����������*
parallel_iterations �
efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Wfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2Stack/TensorListStackTensorListStackCfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/while:output:3nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0*
num_elements��
Jfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Lfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Dfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_3StridedSlice`functional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2Stack/TensorListStack:tensor:0Sfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_3/stack:output:0Ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_3/stack_1:output:0Ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
Efunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
@functional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose_2	Transpose`functional_27_1/bidirectional_12_1/forward_lstm_12_1/TensorArrayV2Stack/TensorListStack:tensor:0Nfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose_2/perm:output:0*
T0*-
_output_shapes
:������������
;functional_27_1/bidirectional_12_1/backward_lstm_12_1/ShapeShape#functional_27_1/masking_8_1/mul:z:0*
T0*
_output_shapes
::���
Ifunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Cfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_sliceStridedSliceDfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/Shape:output:0Rfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice/stack:output:0Tfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice/stack_1:output:0Tfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
Bfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros/packedPackLfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice:output:0Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:�
Afunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
;functional_27_1/bidirectional_12_1/backward_lstm_12_1/zerosFillKfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros/packed:output:0Jfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros/Const:output:0*
T0*(
_output_shapes
:�����������
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_1/packedPackLfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice:output:0Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:�
Cfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
=functional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_1FillMfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_1/packed:output:0Lfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_1/Const:output:0*
T0*(
_output_shapes
:�����������
Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
Efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_1StridedSlice#functional_27_1/masking_8_1/mul:z:0Tfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_1/stack:output:0Vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_1/stack_1:output:0Vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
?functional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose	Transpose#functional_27_1/masking_8_1/mul:z:0Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose/perm:output:0*
T0*,
_output_shapes
:�����������
Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
@functional_27_1/bidirectional_12_1/backward_lstm_12_1/ExpandDims
ExpandDimsfunctional_27_1/Any:output:0Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:�����������
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
Afunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose_1	TransposeIfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ExpandDims:output:0Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose_1/perm:output:0*
T0
*,
_output_shapes
:�����������
Qfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Pfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
Cfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2TensorListReserveZfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2/element_shape:output:0Yfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
?functional_27_1/bidirectional_12_1/backward_lstm_12_1/ReverseV2	ReverseV2Cfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose:y:0Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ReverseV2/axis:output:0*
T0*,
_output_shapes
:�����������
kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
]functional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorHfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ReverseV2:output:0tfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_2StridedSliceCfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose:y:0Tfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_2/stack:output:0Vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_2/stack_1:output:0Vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
Ufunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOpReadVariableOp^functional_27_1_bidirectional_12_1_backward_lstm_12_1_lstm_cell_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
Hfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/MatMulMatMulNfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_2:output:0]functional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Wfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOp`functional_27_1_bidirectional_12_1_backward_lstm_12_1_lstm_cell_1_cast_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
Jfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/MatMul_1MatMulDfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros:output:0_functional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/addAddV2Rfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/MatMul:product:0Tfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
Vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOpReadVariableOp_functional_27_1_bidirectional_12_1_backward_lstm_12_1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Gfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_1AddV2Ifunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add:z:0^functional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Qfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Gfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/splitSplitZfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/split/split_dim:output:0Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_1:z:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
Ifunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/SigmoidSigmoidPfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:�����������
Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Sigmoid_1SigmoidPfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:�����������
Efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/mulMulOfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Sigmoid_1:y:0Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_1:output:0*
T0*(
_output_shapes
:�����������
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/TanhTanhPfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:�����������
Gfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/mul_1MulMfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Sigmoid:y:0Jfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:�����������
Gfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_2AddV2Ifunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/mul:z:0Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/mul_1:z:0*
T0*(
_output_shapes
:�����������
Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Sigmoid_2SigmoidPfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:�����������
Hfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Tanh_1TanhKfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_2:z:0*
T0*(
_output_shapes
:�����������
Gfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/mul_2MulOfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Sigmoid_2:y:0Lfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:�����������
Sfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Rfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
Efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_1TensorListReserve\functional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_1/element_shape:output:0[functional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���|
:functional_27_1/bidirectional_12_1/backward_lstm_12_1/timeConst*
_output_shapes
: *
dtype0*
value	B : �
@functional_27_1/bidirectional_12_1/backward_lstm_12_1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value
B :�|
:functional_27_1/bidirectional_12_1/backward_lstm_12_1/RankConst*
_output_shapes
: *
dtype0*
value	B : �
Afunctional_27_1/bidirectional_12_1/backward_lstm_12_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : �
Afunctional_27_1/bidirectional_12_1/backward_lstm_12_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
;functional_27_1/bidirectional_12_1/backward_lstm_12_1/rangeRangeJfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/range/start:output:0Cfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/Rank:output:0Jfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/range/delta:output:0*
_output_shapes
: �
?functional_27_1/bidirectional_12_1/backward_lstm_12_1/Max/inputConst*
_output_shapes
: *
dtype0*
value
B :��
9functional_27_1/bidirectional_12_1/backward_lstm_12_1/MaxMaxHfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/Max/input:output:0Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/range:output:0*
T0*
_output_shapes
: �
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ReverseV2_1/axisConst*
_output_shapes
:*
dtype0*
valueB: �
Afunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ReverseV2_1	ReverseV2Efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose_1:y:0Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ReverseV2_1/axis:output:0*
T0
*,
_output_shapes
:�����������
Sfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
Rfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
Efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_2TensorListReserve\functional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_2/element_shape:output:0[functional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
_functional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorJfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/ReverseV2_1:output:0vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
@functional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_like	ZerosLikeKfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:�����������
Hfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
;functional_27_1/bidirectional_12_1/backward_lstm_12_1/whileWhileQfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while/loop_counter:output:0Bfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/Max:output:0Cfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/time:output:0Nfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2_1:handle:0Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_like:y:0Dfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros:output:0Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_1:output:0mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0^functional_27_1_bidirectional_12_1_backward_lstm_12_1_lstm_cell_1_cast_readvariableop_resource`functional_27_1_bidirectional_12_1_backward_lstm_12_1_lstm_cell_1_cast_1_readvariableop_resource_functional_27_1_bidirectional_12_1_backward_lstm_12_1_lstm_cell_1_add_1_readvariableop_resourceDfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/zeros_like:y:0*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*v
_output_shapesd
b: : : : :����������:����������:����������: : : : : :����������*%
_read_only_resource_inputs
	
*R
bodyJRH
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_body_84963*R
condJRH
Ffunctional_27_1_bidirectional_12_1_backward_lstm_12_1_while_cond_84962*u
output_shapesd
b: : : : :����������:����������:����������: : : : : :����������*
parallel_iterations �
ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Xfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2Stack/TensorListStackTensorListStackDfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/while:output:3ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:�����������*
element_dtype0*
num_elements��
Kfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Mfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_3StridedSliceafunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2Stack/TensorListStack:tensor:0Tfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_3/stack:output:0Vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_3/stack_1:output:0Vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
Ffunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
Afunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose_2	Transposeafunctional_27_1/bidirectional_12_1/backward_lstm_12_1/TensorArrayV2Stack/TensorListStack:tensor:0Ofunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose_2/perm:output:0*
T0*-
_output_shapes
:�����������{
1functional_27_1/bidirectional_12_1/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
,functional_27_1/bidirectional_12_1/ReverseV2	ReverseV2Efunctional_27_1/bidirectional_12_1/backward_lstm_12_1/transpose_2:y:0:functional_27_1/bidirectional_12_1/ReverseV2/axis:output:0*
T0*-
_output_shapes
:�����������y
.functional_27_1/bidirectional_12_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
)functional_27_1/bidirectional_12_1/concatConcatV2Dfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/transpose_2:y:05functional_27_1/bidirectional_12_1/ReverseV2:output:07functional_27_1/bidirectional_12_1/concat/axis:output:0*
N*
T0*-
_output_shapes
:������������
-functional_27_1/dense_9_1/Cast/ReadVariableOpReadVariableOp6functional_27_1_dense_9_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
 functional_27_1/dense_9_1/MatMulBatchMatMulV22functional_27_1/bidirectional_12_1/concat:output:05functional_27_1/dense_9_1/Cast/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
0functional_27_1/dense_9_1/BiasAdd/ReadVariableOpReadVariableOp9functional_27_1_dense_9_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!functional_27_1/dense_9_1/BiasAddBiasAdd)functional_27_1/dense_9_1/MatMul:output:08functional_27_1/dense_9_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
!functional_27_1/dense_9_1/SoftmaxSoftmax*functional_27_1/dense_9_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������
IdentityIdentity+functional_27_1/dense_9_1/Softmax:softmax:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOpV^functional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOpX^functional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOpW^functional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOp<^functional_27_1/bidirectional_12_1/backward_lstm_12_1/whileU^functional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOpW^functional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOpV^functional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOp;^functional_27_1/bidirectional_12_1/forward_lstm_12_1/while1^functional_27_1/dense_9_1/BiasAdd/ReadVariableOp.^functional_27_1/dense_9_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : : : 2�
Ufunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOpUfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOp2�
Wfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOpWfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOp2�
Vfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOpVfunctional_27_1/bidirectional_12_1/backward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOp2z
;functional_27_1/bidirectional_12_1/backward_lstm_12_1/while;functional_27_1/bidirectional_12_1/backward_lstm_12_1/while2�
Tfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOpTfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast/ReadVariableOp2�
Vfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOpVfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/Cast_1/ReadVariableOp2�
Ufunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOpUfunctional_27_1/bidirectional_12_1/forward_lstm_12_1/lstm_cell_1/add_1/ReadVariableOp2x
:functional_27_1/bidirectional_12_1/forward_lstm_12_1/while:functional_27_1/bidirectional_12_1/forward_lstm_12_1/while2d
0functional_27_1/dense_9_1/BiasAdd/ReadVariableOp0functional_27_1/dense_9_1/BiasAdd/ReadVariableOp2^
-functional_27_1/dense_9_1/Cast/ReadVariableOp-functional_27_1/dense_9_1/Cast/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
,
_output_shapes
:����������
$
_user_specified_name
input_data
�X
�
!__inference__traced_restore_85355
file_prefix.
assignvariableop_variable_9:	�1
assignvariableop_1_variable_8:
��,
assignvariableop_2_variable_7:	�+
assignvariableop_3_variable_6:	0
assignvariableop_4_variable_5:	�1
assignvariableop_5_variable_4:
��,
assignvariableop_6_variable_3:	�+
assignvariableop_7_variable_2:	0
assignvariableop_8_variable_1:	�)
assignvariableop_9_variable:Z
Gassignvariableop_10_bidirectional_12_forward_lstm_12_lstm_cell_kernel_1:	�T
Eassignvariableop_11_bidirectional_12_forward_lstm_12_lstm_cell_bias_1:	�[
Hassignvariableop_12_bidirectional_12_backward_lstm_12_lstm_cell_kernel_1:	�f
Rassignvariableop_13_bidirectional_12_backward_lstm_12_lstm_cell_recurrent_kernel_1:
��7
$assignvariableop_14_dense_9_kernel_1:	�0
"assignvariableop_15_dense_9_bias_1:e
Qassignvariableop_16_bidirectional_12_forward_lstm_12_lstm_cell_recurrent_kernel_1:
��U
Fassignvariableop_17_bidirectional_12_backward_lstm_12_lstm_cell_bias_1:	�
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_9Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_8Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_7Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_6Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_5Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_4Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_3Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_2Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variableIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpGassignvariableop_10_bidirectional_12_forward_lstm_12_lstm_cell_kernel_1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpEassignvariableop_11_bidirectional_12_forward_lstm_12_lstm_cell_bias_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpHassignvariableop_12_bidirectional_12_backward_lstm_12_lstm_cell_kernel_1Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpRassignvariableop_13_bidirectional_12_backward_lstm_12_lstm_cell_recurrent_kernel_1Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp$assignvariableop_14_dense_9_kernel_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_9_bias_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpQassignvariableop_16_bidirectional_12_forward_lstm_12_lstm_cell_recurrent_kernel_1Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpFassignvariableop_17_bidirectional_12_backward_lstm_12_lstm_cell_bias_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_19Identity_19:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:RN
L
_user_specified_name42bidirectional_12/backward_lstm_12/lstm_cell/bias_1:]Y
W
_user_specified_name?=bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel_1:.*
(
_user_specified_namedense_9/bias_1:0,
*
_user_specified_namedense_9/kernel_1:^Z
X
_user_specified_name@>bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel_1:TP
N
_user_specified_name64bidirectional_12/backward_lstm_12/lstm_cell/kernel_1:QM
K
_user_specified_name31bidirectional_12/forward_lstm_12/lstm_cell/bias_1:SO
M
_user_specified_name53bidirectional_12/forward_lstm_12/lstm_cell/kernel_1:(
$
"
_user_specified_name
Variable:*	&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
<

input_data.
serve_input_data:0����������A
output_05
StatefulPartitionedCall:0����������tensorflow/serving/predict*�
serving_default�
F

input_data8
serving_default_input_data:0����������C
output_07
StatefulPartitionedCall_1:0����������tensorflow/serving/predict:�
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
X
0
	1

2
3
4
5
6
7"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trace_02�
__inference___call___85077�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&

input_data����������ztrace_0
7
	serve
serving_default"
signature_map
D:B	�21bidirectional_12/forward_lstm_12/lstm_cell/kernel
O:M
��2;bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel
>:<�2/bidirectional_12/forward_lstm_12/lstm_cell/bias
2:0	2&seed_generator_41/seed_generator_state
E:C	�22bidirectional_12/backward_lstm_12/lstm_cell/kernel
P:N
��2<bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel
?:=�20bidirectional_12/backward_lstm_12/lstm_cell/bias
2:0	2&seed_generator_42/seed_generator_state
!:	�2dense_9/kernel
:2dense_9/bias
D:B	�21bidirectional_12/forward_lstm_12/lstm_cell/kernel
>:<�2/bidirectional_12/forward_lstm_12/lstm_cell/bias
E:C	�22bidirectional_12/backward_lstm_12/lstm_cell/kernel
P:N
��2<bidirectional_12/backward_lstm_12/lstm_cell/recurrent_kernel
!:	�2dense_9/kernel
:2dense_9/bias
O:M
��2;bidirectional_12/forward_lstm_12/lstm_cell/recurrent_kernel
?:=�20bidirectional_12/backward_lstm_12/lstm_cell/bias
�B�
__inference___call___85077
input_data"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_signature_wrapper___call___85099
input_data"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
j
input_data
kwonlydefaults
 
annotations� *
 
�B�
,__inference_signature_wrapper___call___85120
input_data"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
j
input_data
kwonlydefaults
 
annotations� *
 �
__inference___call___85077l	
8�5
.�+
)�&

input_data����������
� "&�#
unknown�����������
,__inference_signature_wrapper___call___85099�	
F�C
� 
<�9
7

input_data)�&

input_data����������"8�5
3
output_0'�$
output_0�����������
,__inference_signature_wrapper___call___85120�	
F�C
� 
<�9
7

input_data)�&

input_data����������"8�5
3
output_0'�$
output_0����������