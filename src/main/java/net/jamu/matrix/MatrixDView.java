/*
 * Copyright 2026 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.jamu.matrix;

import net.frobenius.TTrans;

/**
 * A read-only, live view of a rectangular region of another {@code MatrixD}.
 */
final class MatrixDView extends DimensionsBase implements MatrixD {

    private static final String READ_ONLY = "a matrix view is read-only";

    private final MatrixD parent;
    private final int r0;
    private final int c0;

    MatrixDView(MatrixD parent, int r0, int c0, int rows, int cols) {
        super(rows, cols, false, Double.TYPE);
        this.parent = parent;
        this.r0 = r0;
        this.c0 = c0;
    }

    static MatrixD create(MatrixD A, int r0, int c0, int r1, int c1) {
        if (A == null) {
            throw new NullPointerException("A");
        }
        A.checkSubmatrixIndexes(r0, c0, r1, c1);
        if (A instanceof MatrixDView) {
            MatrixDView v = (MatrixDView) A;
            return new MatrixDView(v.parent, v.r0 + r0, v.c0 + c0, r1 - r0 + 1, c1 - c0 + 1);
        }
        return new MatrixDView(A, r0, c0, r1 - r0 + 1, c1 - c0 + 1);
    }

    /**
     * Returns whether this view reads from {@code array}.
     */
    boolean shares(double[] array) {
        return parent.getArrayUnsafe() == array;
    }

    /**
     * Returns this view as a gemm operand, read from a copy if its parent is
     * {@code out}.
     */
    ReadAccess.OperandD operand(double[] out) {
        double[] p = parent.getArrayUnsafe();
        if (p == out) {
            return new ReadAccess.OperandD(copy().getArrayUnsafe(), 0, rows);
        }
        // the region lies within the parent, so the last index read,
        // (c0 + cols - 1) * ld + r0 + rows - 1, is within p, and ld >= rows
        int ld = parent.numRows();
        return new ReadAccess.OperandD(p, c0 * ld + r0, ld);
    }

    /**
     * Throws if {@code m} is a view.
     *
     * @throws UnsupportedOperationException
     *             if {@code m} is a view
     */
    static void refuse(MatrixD m) {
        if (m instanceof MatrixDView) {
            throw readOnly();
        }
    }

    private static UnsupportedOperationException readOnly() {
        return new UnsupportedOperationException(READ_ONLY);
    }

    // direct reads

    @Override
    public double get(int row, int col) {
        checkIndex(row, col);
        return parent.getUnsafe(r0 + row, c0 + col);
    }

    @Override
    public double getUnsafe(int row, int col) {
        return parent.getUnsafe(r0 + row, c0 + col);
    }

    @Override
    public double toScalar() {
        if (!isScalar()) {
            throw new IllegalStateException("(" + rows + " x " + cols + ") matrix is not a scalar");
        }
        return parent.getUnsafe(r0, c0);
    }

    @Override
    public MatrixD copy() {
        MatrixD C = Matrices.createD(rows, cols);
        return parent.submatrix(r0, c0, r0 + rows - 1, c0 + cols - 1, C, 0, 0);
    }

    @Override
    public String toString() {
        return Matrices.toString(this);
    }

    // mutators

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD scaleInplace(double alpha) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD addInplace(MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD addInplace(double alpha, MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD zeroInplace() {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD setInplace(MatrixD other) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD setInplace(double alpha, MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD setColumnInplace(int colIdx, MatrixD colVector) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD setInplaceUpperTrapezoidal(MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD setInplaceLowerTrapezoidal(MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD set(int row, int col, double val) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD add(int row, int col, double val) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD setSubmatrixInplace(int r0, int c0, MatrixD B, int rb0, int cb0, int rb1, int cb1) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD clampInplace(double min, double max) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD mapInplace(DFunction f) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD addBroadcastedVectorInplace(MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD mulBroadcastedVectorInplace(MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD divBroadcastedVectorInplace(MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD addBroadcastedRowVectorInplace(MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD mulBroadcastedRowVectorInplace(MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD divBroadcastedRowVectorInplace(MatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD zeroizeSubEpsilonInplace(int k) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD zeroizeSubEpsilonRelativeInplace(int k) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD sanitizeNonFiniteInplace(double nanSurrogate, double posInfSurrogate, double negInfSurrogate) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixD sanitizeNaNInplace(double nanSurrogate) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public void setUnsafe(int row, int col, double val) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public double[] getArrayUnsafe() {
        throw readOnly();
    }

    // reads on a copy

    @Override
    public MatrixD scale(double alpha, MatrixD B) {
        return copy().scale(alpha, B);
    }

    @Override
    public MatrixD trans(MatrixD AT) {
        return copy().trans(AT);
    }

    @Override
    public MatrixD add(MatrixD B, MatrixD C) {
        return copy().add(B, C);
    }

    @Override
    public MatrixD add(double alpha, MatrixD B, MatrixD C) {
        return copy().add(alpha, B, C);
    }

    @Override
    public MatrixD solve(MatrixD B, MatrixD X) {
        Checks.checkSolve(this, B, X);
        // the copy is the LAPACK work array, so the solver must not copy again
        double[] work = copy().getArrayUnsafe();
        if (isSquareMatrix()) {
            return SimpleMatrixD.lusolve(work, rows, X, B);
        }
        return SimpleMatrixD.qrsolve(work, rows, cols, X, B);
    }

    @Override
    public MatrixD inv(MatrixD inverse) {
        if (!isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        Checks.checkEqualDimension(this, inverse);
        return solve(Matrices.identityD(rows), inverse);
    }

    @Override
    public MatrixD pseudoInv() {
        return copy().pseudoInv();
    }

    @Override
    public MatrixD expm() {
        return copy().expm();
    }

    @Override
    public MatrixD hadamard(MatrixD B, MatrixD out) {
        return copy().hadamard(B, out);
    }

    @Override
    public double[][] toJaggedArray() {
        return copy().toJaggedArray();
    }

    @Override
    public double normF() {
        return copy().normF();
    }

    @Override
    public double normMaxAbs() {
        return copy().normMaxAbs();
    }

    @Override
    public double normInf() {
        return copy().normInf();
    }

    @Override
    public double norm1() {
        return copy().norm1();
    }

    @Override
    public double trace() {
        return copy().trace();
    }

    @Override
    public MatrixD selectColumn(int col) {
        return copy().selectColumn(col);
    }

    @Override
    public MatrixD selectConsecutiveColumns(int colFrom, int colTo) {
        return copy().selectConsecutiveColumns(colFrom, colTo);
    }

    @Override
    public MatrixD selectSubmatrix(int rowFrom, int colFrom, int rowTo, int colTo) {
        return copy().selectSubmatrix(rowFrom, colFrom, rowTo, colTo);
    }

    @Override
    public MatrixD appendColumn(MatrixD colVector) {
        return copy().appendColumn(colVector);
    }

    @Override
    public MatrixD appendMatrix(MatrixD matrix) {
        return copy().appendMatrix(matrix);
    }

    @Override
    public MatrixD mldivide(MatrixD B) {
        Checks.checkSameRows(this, B);
        return solve(B, Matrices.createD(cols, B.numColumns()));
    }

    @Override
    public MatrixD mrdivide(MatrixD B) {
        return copy().mrdivide(B);
    }

    @Override
    public MatrixD timesMany(MatrixD m, MatrixD... matrices) {
        // two and three factors need no chain and so no copy
        if (matrices.length == 0) {
            return times(m);
        }
        if (matrices.length == 1) {
            return timesTimes(m, matrices[0]);
        }
        return copy().timesMany(m, matrices);
    }

    @Override
    public ComplexMatrixD times(ComplexMatrixD B) {
        return copy().times(B);
    }

    @Override
    public MatrixD plus(MatrixD B) {
        return copy().plus(B);
    }

    @Override
    public MatrixD minus(MatrixD B) {
        return copy().minus(B);
    }

    @Override
    public MatrixD uminus() {
        return copy().uminus();
    }

    @Override
    public MatrixD abs() {
        return copy().abs();
    }

    @Override
    public MatrixD transpose() {
        return copy().transpose();
    }

    @Override
    public MatrixD inverse() {
        if (!isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        return inv(Matrices.createD(rows, cols));
    }

    @Override
    public MatrixD hadamard(MatrixD B) {
        return copy().hadamard(B);
    }

    @Override
    public MatrixD hadamardTransposed(MatrixD B) {
        return copy().hadamardTransposed(B);
    }

    @Override
    public MatrixD transposedHadamard(MatrixD B) {
        return copy().transposedHadamard(B);
    }

    @Override
    public MatrixD map(DFunction f) {
        return copy().map(f);
    }

    @Override
    public MatrixD plusBroadcastedVector(MatrixD B) {
        return copy().plusBroadcastedVector(B);
    }

    @Override
    public MatrixD mulBroadcastedVector(MatrixD B) {
        return copy().mulBroadcastedVector(B);
    }

    @Override
    public MatrixD divBroadcastedVector(MatrixD B) {
        return copy().divBroadcastedVector(B);
    }

    @Override
    public MatrixD plusBroadcastedRowVector(MatrixD B) {
        return copy().plusBroadcastedRowVector(B);
    }

    @Override
    public MatrixD mulBroadcastedRowVector(MatrixD B) {
        return copy().mulBroadcastedRowVector(B);
    }

    @Override
    public MatrixD divBroadcastedRowVector(MatrixD B) {
        return copy().divBroadcastedRowVector(B);
    }

    @Override
    public MatrixD reshape(int rows, int cols) {
        return copy().reshape(rows, cols);
    }

    @Override
    public ComplexMatrixD toComplexMatrix() {
        return copy().toComplexMatrix();
    }

    // products through gemm on the parent

    @Override
    public MatrixD mult(MatrixD B, MatrixD C) {
        return mult(1.0, B, C);
    }

    @Override
    public MatrixD mult(double alpha, MatrixD B, MatrixD C) {
        if (shares(C.getArrayUnsafe())) {
            // zeroing the output would zero this view
            return copy().mult(alpha, B, C);
        }
        return multAdd(alpha, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public MatrixD multAdd(MatrixD B, MatrixD C) {
        return multAdd(1.0, B, C);
    }

    @Override
    public MatrixD multAdd(double alpha, MatrixD B, MatrixD C) {
        Checks.checkMultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD a = operand(c);
        ReadAccess.OperandD b = ReadAccess.operand(B, c);
        Matrices.getBlas().dgemm(TTrans.NO_TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), cols,
                alpha, a.array, a.offset, a.ld, b.array, b.offset, b.ld, 1.0, c, 0, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public MatrixD transABmult(MatrixD B, MatrixD C) {
        return transABmult(1.0, B, C);
    }

    @Override
    public MatrixD transABmult(double alpha, MatrixD B, MatrixD C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().transABmult(alpha, B, C);
        }
        return transABmultAdd(alpha, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public MatrixD transABmultAdd(MatrixD B, MatrixD C) {
        return transABmultAdd(1.0, B, C);
    }

    @Override
    public MatrixD transABmultAdd(double alpha, MatrixD B, MatrixD C) {
        Checks.checkTransABmultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD a = operand(c);
        ReadAccess.OperandD b = ReadAccess.operand(B, c);
        Matrices.getBlas().dgemm(TTrans.TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), rows, alpha,
                a.array, a.offset, a.ld, b.array, b.offset, b.ld, 1.0, c, 0, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public MatrixD transAmult(MatrixD B, MatrixD C) {
        return transAmult(1.0, B, C);
    }

    @Override
    public MatrixD transAmult(double alpha, MatrixD B, MatrixD C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().transAmult(alpha, B, C);
        }
        return transAmultAdd(alpha, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public MatrixD transAmultAdd(MatrixD B, MatrixD C) {
        return transAmultAdd(1.0, B, C);
    }

    @Override
    public MatrixD transAmultAdd(double alpha, MatrixD B, MatrixD C) {
        Checks.checkTransAmultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD a = operand(c);
        ReadAccess.OperandD b = ReadAccess.operand(B, c);
        Matrices.getBlas().dgemm(TTrans.TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), rows, alpha,
                a.array, a.offset, a.ld, b.array, b.offset, b.ld, 1.0, c, 0, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public MatrixD transBmult(MatrixD B, MatrixD C) {
        return transBmult(1.0, B, C);
    }

    @Override
    public MatrixD transBmult(double alpha, MatrixD B, MatrixD C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().transBmult(alpha, B, C);
        }
        return transBmultAdd(alpha, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public MatrixD transBmultAdd(MatrixD B, MatrixD C) {
        return transBmultAdd(1.0, B, C);
    }

    @Override
    public MatrixD transBmultAdd(double alpha, MatrixD B, MatrixD C) {
        Checks.checkTransBmultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD a = operand(c);
        ReadAccess.OperandD b = ReadAccess.operand(B, c);
        Matrices.getBlas().dgemm(TTrans.NO_TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), cols, alpha,
                a.array, a.offset, a.ld, b.array, b.offset, b.ld, 1.0, c, 0, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public MatrixD times(MatrixD B) {
        return mult(B, Matrices.createD(rows, B.numColumns()));
    }

    @Override
    public MatrixD timesTimes(MatrixD B, MatrixD C) {
        if (Matrices.aTimesBfirst(this, B, C)) {
            return mult(B, Matrices.createD(rows, B.numColumns())).mult(C, Matrices.createD(rows, C.numColumns()));
        }
        return mult(B.mult(C, Matrices.createD(B.numRows(), C.numColumns())), Matrices.createD(rows, C.numColumns()));
    }

    @Override
    public MatrixD timesTransposed() {
        return transBmult(this, Matrices.createD(rows, rows));
    }

    @Override
    public MatrixD timesTransposed(MatrixD B) {
        return transBmult(B, Matrices.createD(rows, B.numRows()));
    }

    @Override
    public MatrixD transposedTimes() {
        return transAmult(this, Matrices.createD(cols, cols));
    }

    @Override
    public MatrixD transposedTimes(MatrixD B) {
        return transAmult(B, Matrices.createD(cols, B.numColumns()));
    }

    @Override
    public MatrixD timesPlus(MatrixD B, MatrixD C) {
        return multAdd(B, C.copy());
    }

    @Override
    public MatrixD timesMinus(MatrixD B, MatrixD C) {
        return multAdd(B, C.uminus());
    }

    // reads on the parent or decompositions that copy their input

    @Override
    public MatrixD submatrix(int row0, int col0, int row1, int col1, MatrixD B, int rb, int cb) {
        // the parent only knows its own, larger bounds
        checkSubmatrixIndexes(row0, col0, row1, col1);
        return parent.submatrix(r0 + row0, c0 + col0, r0 + row1, c0 + col1, B, rb, cb);
    }

    @Override
    public SvdD svd(boolean full) {
        return new SvdD(this, full);
    }

    @Override
    public SvdEconD svdEcon() {
        return new SvdEconD(this);
    }

    @Override
    public double[] singularValues() {
        return svd(false).getS();
    }

    @Override
    public EvdD evd(boolean full) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("EVD only works for square matrices");
        }
        return new EvdD(this, full);
    }

    @Override
    public QrdD qrd() {
        return new QrdD(this);
    }

    @Override
    public LudD lud() {
        return new LudD(this);
    }

    @Override
    public double norm2() {
        return new SvdD(this, false).norm2();
    }
}
