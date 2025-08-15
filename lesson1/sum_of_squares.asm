	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0
	.globl	__ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE
	.p2align	2
__ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE:
	cmp	x7, #1
	b.lt	LBB0_3
	ldr	x8, [sp]
	cmp	x7, #1
	b.ne	LBB0_4
	mov	x9, #0
	movi	d0, #0000000000000000
	b	LBB0_7
LBB0_3:
	movi	d0, #0000000000000000
	b	LBB0_9
LBB0_4:
	and	x9, x7, #0xfffffffffffffffe
	lsl	x10, x8, #1
	neg	x11, x9
	movi	d0, #0000000000000000
	mov	x12, x6
LBB0_5:
	ldr	d1, [x12]
	ldr	d2, [x8, x12]
	fmul	d1, d1, d1
	fmul	d2, d2, d2
	fadd	d0, d0, d1
	fadd	d0, d0, d2
	add	x12, x12, x10
	adds	x11, x11, #2
	b.ne	LBB0_5
	cmp	x9, x7
	b.eq	LBB0_9
LBB0_7:
	madd	x10, x9, x8, x6
	sub	x9, x9, x7
LBB0_8:
	ldr	d1, [x10]
	fmul	d1, d1, d1
	fadd	d0, d0, d1
	add	x10, x10, x8
	adds	x9, x9, #1
	b.lo	LBB0_8
LBB0_9:
	str	d0, [x0]
	mov	w0, #0
	ret

	.globl	__ZN7cpython8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE
	.p2align	2
__ZN7cpython8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE:
	.cfi_startproc
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
	stp	d9, d8, [sp, #80]
	stp	x20, x19, [sp, #96]
	stp	x29, x30, [sp, #112]
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	mov	x0, x1
	add	x8, sp, #72
	str	x8, [sp]
Lloh0:
	adrp	x1, _.const.sum_of_squares_numba@GOTPAGE
Lloh1:
	ldr	x1, [x1, _.const.sum_of_squares_numba@GOTPAGEOFF]
Lloh2:
	adrp	x8, _PyArg_UnpackTuple@GOTPAGE
Lloh3:
	ldr	x8, [x8, _PyArg_UnpackTuple@GOTPAGEOFF]
	mov	w2, #1
	mov	w3, #1
	blr	x8
	cbz	w0, LBB1_7
Lloh4:
	adrp	x8, __ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE@GOTPAGE
Lloh5:
	ldr	x8, [x8, __ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE@GOTPAGEOFF]
Lloh6:
	ldr	x8, [x8]
	cbz	x8, LBB1_4
	ldr	x0, [sp, #72]
	movi.2d	v0, #0000000000000000
	stp	q0, q0, [sp, #16]
	str	q0, [sp, #48]
	str	xzr, [sp, #64]
Lloh7:
	adrp	x8, _NRT_adapt_ndarray_from_python@GOTPAGE
Lloh8:
	ldr	x8, [x8, _NRT_adapt_ndarray_from_python@GOTPAGEOFF]
	add	x1, sp, #16
	blr	x8
	ldr	x8, [sp, #40]
	cmp	w0, #0
	ccmp	x8, #8, #0, eq
	b.ne	LBB1_5
	ldr	x19, [sp, #16]
	ldp	x6, x7, [sp, #48]
	ldr	x8, [sp, #64]
	stp	x8, xzr, [sp]
Lloh9:
	adrp	x8, __ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE@GOTPAGE
Lloh10:
	ldr	x8, [x8, __ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE@GOTPAGEOFF]
	add	x0, sp, #8
	blr	x8
	ldr	d8, [sp, #8]
Lloh11:
	adrp	x8, _NRT_decref@GOTPAGE
Lloh12:
	ldr	x8, [x8, _NRT_decref@GOTPAGEOFF]
	mov	x0, x19
	blr	x8
Lloh13:
	adrp	x8, _PyFloat_FromDouble@GOTPAGE
Lloh14:
	ldr	x8, [x8, _PyFloat_FromDouble@GOTPAGEOFF]
	fmov	d0, d8
	blr	x8
	ldp	x29, x30, [sp, #112]
	ldp	x20, x19, [sp, #96]
	ldp	d9, d8, [sp, #80]
	add	sp, sp, #128
	ret
LBB1_4:
Lloh15:
	adrp	x0, _PyExc_RuntimeError@GOTPAGE
Lloh16:
	ldr	x0, [x0, _PyExc_RuntimeError@GOTPAGEOFF]
Lloh17:
	adrp	x1, "_.const.missing Environment: _ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE"@GOTPAGE
Lloh18:
	ldr	x1, [x1, "_.const.missing Environment: _ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE"@GOTPAGEOFF]
	b	LBB1_6
LBB1_5:
Lloh19:
	adrp	x0, _PyExc_TypeError@GOTPAGE
Lloh20:
	ldr	x0, [x0, _PyExc_TypeError@GOTPAGEOFF]
Lloh21:
	adrp	x1, "_.const.can't unbox array from PyObject into native value.  The object maybe of a different type"@GOTPAGE
Lloh22:
	ldr	x1, [x1, "_.const.can't unbox array from PyObject into native value.  The object maybe of a different type"@GOTPAGEOFF]
LBB1_6:
Lloh23:
	adrp	x8, _PyErr_SetString@GOTPAGE
Lloh24:
	ldr	x8, [x8, _PyErr_SetString@GOTPAGEOFF]
	blr	x8
LBB1_7:
	mov	x0, #0
	ldp	x29, x30, [sp, #112]
	ldp	x20, x19, [sp, #96]
	ldp	d9, d8, [sp, #80]
	add	sp, sp, #128
	ret
	.loh AdrpLdrGot	Lloh2, Lloh3
	.loh AdrpLdrGot	Lloh0, Lloh1
	.loh AdrpLdrGotLdr	Lloh4, Lloh5, Lloh6
	.loh AdrpLdrGot	Lloh7, Lloh8
	.loh AdrpLdrGot	Lloh13, Lloh14
	.loh AdrpLdrGot	Lloh11, Lloh12
	.loh AdrpLdrGot	Lloh9, Lloh10
	.loh AdrpLdrGot	Lloh17, Lloh18
	.loh AdrpLdrGot	Lloh15, Lloh16
	.loh AdrpLdrGot	Lloh21, Lloh22
	.loh AdrpLdrGot	Lloh19, Lloh20
	.loh AdrpLdrGot	Lloh23, Lloh24
	.cfi_endproc

	.globl	_cfunc._ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE
	.p2align	2
_cfunc._ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE:
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]
	stp	x6, xzr, [sp]
Lloh25:
	adrp	x8, __ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE@GOTPAGE
Lloh26:
	ldr	x8, [x8, __ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE@GOTPAGEOFF]
	add	x0, sp, #8
	mov	x6, x4
	mov	x7, x5
	blr	x8
	ldr	d0, [sp, #8]
	ldp	x29, x30, [sp, #16]
	add	sp, sp, #32
	ret
	.loh AdrpLdrGot	Lloh25, Lloh26

	.globl	_NRT_decref
	.weak_def_can_be_hidden	_NRT_decref
	.p2align	2
_NRT_decref:
	.cfi_startproc
	cbz	x0, LBB3_2
	dmb	ish
	mov	x8, #-1
	ldadd	x8, x8, [x0]
	cmp	x8, #1
	b.eq	LBB3_3
LBB3_2:
	ret
LBB3_3:
	dmb	ishld
Lloh27:
	adrp	x1, _NRT_MemInfo_call_dtor@GOTPAGE
Lloh28:
	ldr	x1, [x1, _NRT_MemInfo_call_dtor@GOTPAGEOFF]
	br	x1
	.loh AdrpLdrGot	Lloh27, Lloh28
	.cfi_endproc

	.section	__TEXT,__const
	.p2align	4
_.const.sum_of_squares_numba:
	.asciz	"sum_of_squares_numba"

	.comm	__ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE,8,3
	.p2align	4
"_.const.missing Environment: _ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE":
	.asciz	"missing Environment: _ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE"

	.p2align	4
"_.const.can't unbox array from PyObject into native value.  The object maybe of a different type":
	.asciz	"can't unbox array from PyObject into native value.  The object maybe of a different type"

.subsections_via_symbols
