; ModuleID = 'sum_of_squares_numba'
source_filename = "<string>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin24.5.0"

@.const.sum_of_squares_numba = internal constant [21 x i8] c"sum_of_squares_numba\00"
@_ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE = common local_unnamed_addr global i8* null
@".const.missing Environment: _ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE" = internal constant [143 x i8] c"missing Environment: _ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE\00"
@PyExc_TypeError = external global i8
@".const.can't unbox array from PyObject into native value.  The object maybe of a different type" = internal constant [89 x i8] c"can't unbox array from PyObject into native value.  The object maybe of a different type\00"
@PyExc_RuntimeError = external global i8

; Function Attrs: nofree norecurse nosync nounwind
define i32 @_ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE(double* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture readnone %excinfo, i8* nocapture readnone %arg.arr.0, i8* nocapture readnone %arg.arr.1, i64 %arg.arr.2, i64 %arg.arr.3, double* %arg.arr.4, i64 %arg.arr.5.0, i64 %arg.arr.6.0) local_unnamed_addr #0 {
entry:
  %.713 = icmp sgt i64 %arg.arr.5.0, 0
  br i1 %.713, label %B12.lr.ph, label %B28

B12.lr.ph:                                        ; preds = %entry
  %min.iters.check = icmp eq i64 %arg.arr.5.0, 1
  br i1 %min.iters.check, label %B12.preheader, label %vector.ph

vector.ph:                                        ; preds = %B12.lr.ph
  %0 = ptrtoint double* %arg.arr.4 to i64
  %n.vec = and i64 %arg.arr.5.0, -2
  %1 = shl i64 %arg.arr.6.0, 1
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv11 = phi i64 [ %lsr.iv.next12, %vector.body ], [ %n.vec, %vector.ph ]
  %lsr.iv9 = phi i64 [ %lsr.iv.next10, %vector.body ], [ %0, %vector.ph ]
  %vec.phi = phi double [ 0.000000e+00, %vector.ph ], [ %10, %vector.body ]
  %2 = add i64 %arg.arr.6.0, %lsr.iv9
  %3 = inttoptr i64 %lsr.iv9 to double*
  %4 = inttoptr i64 %2 to double*
  %5 = load double, double* %3, align 8
  %6 = load double, double* %4, align 8
  %7 = fmul double %5, %5
  %8 = fmul double %6, %6
  %9 = fadd double %vec.phi, %7
  %10 = fadd double %9, %8
  %lsr.iv.next10 = add i64 %lsr.iv9, %1
  %lsr.iv.next12 = add i64 %lsr.iv11, -2
  %11 = icmp eq i64 %lsr.iv.next12, 0
  br i1 %11, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %arg.arr.5.0
  br i1 %cmp.n, label %B28, label %B12.preheader

B12.preheader:                                    ; preds = %B12.lr.ph, %middle.block
  %total.2.05.ph = phi double [ %10, %middle.block ], [ 0.000000e+00, %B12.lr.ph ]
  %.22.04.ph = phi i64 [ %n.vec, %middle.block ], [ 0, %B12.lr.ph ]
  %12 = ptrtoint double* %arg.arr.4 to i64
  %13 = mul i64 %.22.04.ph, %arg.arr.6.0
  %14 = add i64 %12, %13
  %15 = sub i64 %arg.arr.5.0, %.22.04.ph
  br label %B12

B12:                                              ; preds = %B12.preheader, %B12
  %lsr.iv7 = phi i64 [ %15, %B12.preheader ], [ %lsr.iv.next8, %B12 ]
  %lsr.iv = phi i64 [ %14, %B12.preheader ], [ %lsr.iv.next, %B12 ]
  %total.2.05 = phi double [ %.119, %B12 ], [ %total.2.05.ph, %B12.preheader ]
  %.88 = inttoptr i64 %lsr.iv to double*
  %.89 = load double, double* %.88, align 8
  %.117 = fmul double %.89, %.89
  %.119 = fadd double %total.2.05, %.117
  %lsr.iv.next = add i64 %lsr.iv, %arg.arr.6.0
  %lsr.iv.next8 = add i64 %lsr.iv7, -1
  %exitcond.not = icmp eq i64 %lsr.iv.next8, 0
  br i1 %exitcond.not, label %B28, label %B12, !llvm.loop !2

B28:                                              ; preds = %B12, %middle.block, %entry
  %total.2.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %10, %middle.block ], [ %.119, %B12 ]
  store double %total.2.0.lcssa, double* %retptr, align 8
  ret i32 0
}

define i8* @_ZN7cpython8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE(i8* nocapture readnone %py_closure, i8* %py_args, i8* nocapture readnone %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca i8*, align 8
  %.6 = call i32 (i8*, i8*, i64, i64, ...) @PyArg_UnpackTuple(i8* %py_args, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.const.sum_of_squares_numba, i64 0, i64 0), i64 1, i64 1, i8** nonnull %.5)
  %.7 = icmp eq i32 %.6, 0
  %.21 = alloca { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] }, align 8
  %.43 = alloca double, align 8
  br i1 %.7, label %common.ret, label %entry.endif, !prof !3

common.ret:                                       ; preds = %entry.endif.endif.endif.thread, %entry, %entry.endif.endif.endif.endif, %entry.endif.if
  %common.ret.op = phi i8* [ null, %entry.endif.if ], [ %.67, %entry.endif.endif.endif.endif ], [ null, %entry ], [ null, %entry.endif.endif.endif.thread ]
  ret i8* %common.ret.op

entry.endif:                                      ; preds = %entry
  %.11 = load i8*, i8** @_ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE, align 8
  %.16 = icmp eq i8* %.11, null
  br i1 %.16, label %entry.endif.if, label %entry.endif.endif, !prof !3

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([143 x i8], [143 x i8]* @".const.missing Environment: _ZN08NumbaEnv8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE", i64 0, i64 0))
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.20 = load i8*, i8** %.5, align 8
  %.24 = bitcast { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] }* %.21 to i8*
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(56) %.24, i8 0, i64 56, i1 false)
  %.25 = call i32 @NRT_adapt_ndarray_from_python(i8* %.20, i8* nonnull %.24)
  %0 = bitcast { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] }* %.21 to i8*
  %sunkaddr = getelementptr inbounds i8, i8* %0, i64 24
  %1 = bitcast i8* %sunkaddr to i64*
  %.29 = load i64, i64* %1, align 8
  %.30 = icmp ne i64 %.29, 8
  %.31 = icmp ne i32 %.25, 0
  %.32 = or i1 %.31, %.30
  br i1 %.32, label %entry.endif.endif.endif.thread, label %entry.endif.endif.endif.endif, !prof !3

entry.endif.endif.endif.thread:                   ; preds = %entry.endif.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_TypeError, i8* getelementptr inbounds ([89 x i8], [89 x i8]* @".const.can't unbox array from PyObject into native value.  The object maybe of a different type", i64 0, i64 0))
  br label %common.ret

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif
  %2 = bitcast { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] }* %.21 to i8**
  %.36.fca.0.load = load i8*, i8** %2, align 8
  %3 = bitcast { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] }* %.21 to i8*
  %sunkaddr2 = getelementptr inbounds i8, i8* %3, i64 32
  %4 = bitcast i8* %sunkaddr2 to double**
  %.36.fca.4.load = load double*, double** %4, align 8
  %5 = bitcast { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] }* %.21 to i8*
  %sunkaddr3 = getelementptr inbounds i8, i8* %5, i64 40
  %6 = bitcast i8* %sunkaddr3 to i64*
  %.36.fca.5.0.load = load i64, i64* %6, align 8
  %7 = bitcast { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] }* %.21 to i8*
  %sunkaddr4 = getelementptr inbounds i8, i8* %7, i64 48
  %8 = bitcast i8* %sunkaddr4 to i64*
  %.36.fca.6.0.load = load i64, i64* %8, align 8
  store double 0.000000e+00, double* %.43, align 8
  %.49 = call i32 @_ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE(double* nonnull %.43, { i8*, i32, i8*, i8*, i32 }** nonnull poison, i8* poison, i8* poison, i64 poison, i64 poison, double* %.36.fca.4.load, i64 %.36.fca.5.0.load, i64 %.36.fca.6.0.load) #1
  %.59 = load double, double* %.43, align 8
  call void @NRT_decref(i8* %.36.fca.0.load)
  %.67 = call i8* @PyFloat_FromDouble(double %.59)
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(i8*, i8*, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(i8*, i8*) local_unnamed_addr

declare i32 @NRT_adapt_ndarray_from_python(i8* nocapture, i8* nocapture) local_unnamed_addr

declare i8* @PyFloat_FromDouble(double) local_unnamed_addr

; Function Attrs: nofree norecurse nosync nounwind
define double @cfunc._ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE({ i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] } %.1) local_unnamed_addr #0 {
entry:
  %.3 = alloca double, align 8
  store double 0.000000e+00, double* %.3, align 8
  %extracted.data = extractvalue { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] } %.1, 4
  %extracted.shape = extractvalue { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] } %.1, 5
  %.7 = extractvalue [1 x i64] %extracted.shape, 0
  %extracted.strides = extractvalue { i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64] } %.1, 6
  %.8 = extractvalue [1 x i64] %extracted.strides, 0
  %.9 = call i32 @_ZN8__main__20sum_of_squares_numbaB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dE5ArrayIdLi1E1C7mutable7alignedE(double* nonnull %.3, { i8*, i32, i8*, i8*, i32 }** nonnull poison, i8* poison, i8* poison, i64 poison, i64 poison, double* %extracted.data, i64 %.7, i64 %.8) #1
  %.19 = load double, double* %.3, align 8
  ret double %.19
}

; Function Attrs: noinline
define linkonce_odr void @NRT_decref(i8* %.1) local_unnamed_addr #1 {
.3:
  %.4 = icmp eq i8* %.1, null
  br i1 %.4, label %common.ret1, label %.3.endif, !prof !3

common.ret1:                                      ; preds = %.3, %.3.endif
  ret void

.3.endif:                                         ; preds = %.3
  fence release
  %.8 = bitcast i8* %.1 to i64*
  %.4.i = atomicrmw sub i64* %.8, i64 1 monotonic, align 8
  %.10 = icmp eq i64 %.4.i, 1
  br i1 %.10, label %.3.endif.if, label %common.ret1, !prof !3

.3.endif.if:                                      ; preds = %.3.endif
  fence acquire
  tail call void @NRT_MemInfo_call_dtor(i8* nonnull %.1)
  ret void
}

declare void @NRT_MemInfo_call_dtor(i8*) local_unnamed_addr

; Function Attrs: argmemonly nocallback nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #2

attributes #0 = { nofree norecurse nosync nounwind }
attributes #1 = { noinline }
attributes #2 = { argmemonly nocallback nofree nounwind willreturn writeonly }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = distinct !{!2, !1}
!3 = !{!"branch_weights", i32 1, i32 99}
