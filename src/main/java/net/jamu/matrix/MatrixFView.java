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
 * A read-only, live view of a rectangular region of another {@code MatrixF}.
 */
final class MatrixFView extends DimensionsBase implements MatrixF {

    private static final String READ_ONLY = "a matrix view is read-only";

    private final MatrixF parent;
    private final int r0;
    private final int c0;

    MatrixFView(MatrixF parent, int r0, int c0, int rows, int cols) {
        super(rows, cols, false, Float.TYPE);
        this.parent = parent;
        this.r0 = r0;
        this.c0 = c0;
    }

    static MatrixF create(MatrixF A, int r0, int c0, int r1, int c1) {
        if (A == null) {
            throw new NullPointerException("A");
        }
        A.checkSubmatrixIndexes(r0, c0, r1, c1);
        if (A instanceof MatrixFView) {
            MatrixFView v = (MatrixFView) A;
            return new MatrixFView(v.parent, v.r0 + r0, v.c0 + c0, r1 - r0 + 1, c1 - c0 + 1);
        }
        return new MatrixFView(A, r0, c0, r1 - r0 + 1, c1 - c0 + 1);
    }

    /**
     * Returns whether this view reads from {@code array}.
     */
    boolean shares(float[] array) {
        return parent.getArrayUnsafe() == array;
    }

    /**
     * Returns this view as a gemm operand, read from a copy if its parent is
     * {@code out}.
     */
    ReadAccess.OperandF operand(float[] out) {
        float[] p = parent.getArrayUnsafe();
        if (p == out) {
            return new ReadAccess.OperandF(copy().getArrayUnsafe(), 0, rows);
        }
        // the region lies within the parent, so the last index read,
        // (c0 + cols - 1) * ld + r0 + rows - 1, is within p, and ld >= rows
        int ld = parent.numRows();
        return new ReadAccess.OperandF(p, c0 * ld + r0, ld);
    }

    /**
     * Throws if {@code m} is a view.
     *
     * @throws UnsupportedOperationException
     *             if {@code m} is a view
     */
    static void refuse(MatrixF m) {
        if (m instanceof MatrixFView) {
            throw readOnly();
        }
    }

    private static UnsupportedOperationException readOnly() {
        return new UnsupportedOperationException(READ_ONLY);
    }

    // direct reads

    @Override
    public float get(int row, int col) {
        checkIndex(row, col);
        return parent.getUnsafe(r0 + row, c0 + col);
    }

    @Override
    public float getUnsafe(int row, int col) {
        return parent.getUnsafe(r0 + row, c0 + col);
    }

    @Override
    public float toScalar() {
        if (!isScalar()) {
            throw new IllegalStateException("(" + rows + " x " + cols + ") matrix is not a scalar");
        }
        return parent.getUnsafe(r0, c0);
    }

    @Override
    public MatrixF copy() {
        MatrixF C = Matrices.createF(rows, cols);
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
    public MatrixF scaleInplace(float alpha) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF addInplace(MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF addInplace(float alpha, MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF zeroInplace() {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF setInplace(MatrixF other) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF setInplace(float alpha, MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF setColumnInplace(int colIdx, MatrixF colVector) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF setInplaceUpperTrapezoidal(MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF setInplaceLowerTrapezoidal(MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF set(int row, int col, float val) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF add(int row, int col, float val) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF setSubmatrixInplace(int r0, int c0, MatrixF B, int rb0, int cb0, int rb1, int cb1) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF clampInplace(float min, float max) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF mapInplace(FFunction f) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF addBroadcastedVectorInplace(MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF mulBroadcastedVectorInplace(MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF divBroadcastedVectorInplace(MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF addBroadcastedRowVectorInplace(MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF mulBroadcastedRowVectorInplace(MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF divBroadcastedRowVectorInplace(MatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF zeroizeSubEpsilonInplace(int k) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF zeroizeSubEpsilonRelativeInplace(int k) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF sanitizeNonFiniteInplace(float nanSurrogate, float posInfSurrogate, float negInfSurrogate) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public MatrixF sanitizeNaNInplace(float nanSurrogate) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public void setUnsafe(int row, int col, float val) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     * 
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public float[] getArrayUnsafe() {
        throw readOnly();
    }

    // reads on a copy

    @Override
    public MatrixF scale(float alpha, MatrixF B) {
        return copy().scale(alpha, B);
    }

    @Override
    public MatrixF trans(MatrixF AT) {
        return copy().trans(AT);
    }

    @Override
    public MatrixF add(MatrixF B, MatrixF C) {
        return copy().add(B, C);
    }

    @Override
    public MatrixF add(float alpha, MatrixF B, MatrixF C) {
        return copy().add(alpha, B, C);
    }

    @Override
    public MatrixF solve(MatrixF B, MatrixF X) {
        return copy().solve(B, X);
    }

    @Override
    public MatrixF inv(MatrixF inverse) {
        return copy().inv(inverse);
    }

    @Override
    public MatrixF pseudoInv() {
        return copy().pseudoInv();
    }

    @Override
    public MatrixF expm() {
        return copy().expm();
    }

    @Override
    public MatrixF hadamard(MatrixF B, MatrixF out) {
        return copy().hadamard(B, out);
    }

    @Override
    public float[][] toJaggedArray() {
        return copy().toJaggedArray();
    }

    @Override
    public float normF() {
        return copy().normF();
    }

    @Override
    public float normMaxAbs() {
        return copy().normMaxAbs();
    }

    @Override
    public float normInf() {
        return copy().normInf();
    }

    @Override
    public float norm1() {
        return copy().norm1();
    }

    @Override
    public float trace() {
        return copy().trace();
    }

    @Override
    public MatrixF selectColumn(int col) {
        return copy().selectColumn(col);
    }

    @Override
    public MatrixF selectConsecutiveColumns(int colFrom, int colTo) {
        return copy().selectConsecutiveColumns(colFrom, colTo);
    }

    @Override
    public MatrixF selectSubmatrix(int rowFrom, int colFrom, int rowTo, int colTo) {
        return copy().selectSubmatrix(rowFrom, colFrom, rowTo, colTo);
    }

    @Override
    public MatrixF appendColumn(MatrixF colVector) {
        return copy().appendColumn(colVector);
    }

    @Override
    public MatrixF appendMatrix(MatrixF matrix) {
        return copy().appendMatrix(matrix);
    }

    @Override
    public MatrixF mldivide(MatrixF B) {
        return copy().mldivide(B);
    }

    @Override
    public MatrixF mrdivide(MatrixF B) {
        return copy().mrdivide(B);
    }

    @Override
    public MatrixF timesMany(MatrixF m, MatrixF... matrices) {
        return copy().timesMany(m, matrices);
    }

    @Override
    public ComplexMatrixF times(ComplexMatrixF B) {
        return copy().times(B);
    }

    @Override
    public MatrixF plus(MatrixF B) {
        return copy().plus(B);
    }

    @Override
    public MatrixF minus(MatrixF B) {
        return copy().minus(B);
    }

    @Override
    public MatrixF uminus() {
        return copy().uminus();
    }

    @Override
    public MatrixF abs() {
        return copy().abs();
    }

    @Override
    public MatrixF transpose() {
        return copy().transpose();
    }

    @Override
    public MatrixF inverse() {
        return copy().inverse();
    }

    @Override
    public MatrixF hadamard(MatrixF B) {
        return copy().hadamard(B);
    }

    @Override
    public MatrixF hadamardTransposed(MatrixF B) {
        return copy().hadamardTransposed(B);
    }

    @Override
    public MatrixF transposedHadamard(MatrixF B) {
        return copy().transposedHadamard(B);
    }

    @Override
    public MatrixF map(FFunction f) {
        return copy().map(f);
    }

    @Override
    public MatrixF plusBroadcastedVector(MatrixF B) {
        return copy().plusBroadcastedVector(B);
    }

    @Override
    public MatrixF mulBroadcastedVector(MatrixF B) {
        return copy().mulBroadcastedVector(B);
    }

    @Override
    public MatrixF divBroadcastedVector(MatrixF B) {
        return copy().divBroadcastedVector(B);
    }

    @Override
    public MatrixF plusBroadcastedRowVector(MatrixF B) {
        return copy().plusBroadcastedRowVector(B);
    }

    @Override
    public MatrixF mulBroadcastedRowVector(MatrixF B) {
        return copy().mulBroadcastedRowVector(B);
    }

    @Override
    public MatrixF divBroadcastedRowVector(MatrixF B) {
        return copy().divBroadcastedRowVector(B);
    }

    @Override
    public MatrixF reshape(int rows, int cols) {
        return copy().reshape(rows, cols);
    }

    @Override
    public ComplexMatrixF toComplexMatrix() {
        return copy().toComplexMatrix();
    }

    // products through gemm on the parent

    @Override
    public MatrixF mult(MatrixF B, MatrixF C) {
        return mult(1.0f, B, C);
    }

    @Override
    public MatrixF mult(float alpha, MatrixF B, MatrixF C) {
        if (shares(C.getArrayUnsafe())) {
            // zeroing the output would zero this view
            return copy().mult(alpha, B, C);
        }
        return multAdd(alpha, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public MatrixF multAdd(MatrixF B, MatrixF C) {
        return multAdd(1.0f, B, C);
    }

    @Override
    public MatrixF multAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkMultAdd(this, B, C);
        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF a = operand(c);
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        Matrices.getBlas().sgemm(TTrans.NO_TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), cols,
                alpha, a.array, a.offset, a.ld, b.array, b.offset, b.ld, 1.0f, c, 0, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public MatrixF transABmult(MatrixF B, MatrixF C) {
        return transABmult(1.0f, B, C);
    }

    @Override
    public MatrixF transABmult(float alpha, MatrixF B, MatrixF C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().transABmult(alpha, B, C);
        }
        return transABmultAdd(alpha, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public MatrixF transABmultAdd(MatrixF B, MatrixF C) {
        return transABmultAdd(1.0f, B, C);
    }

    @Override
    public MatrixF transABmultAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkTransABmultAdd(this, B, C);
        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF a = operand(c);
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        Matrices.getBlas().sgemm(TTrans.TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), rows, alpha,
                a.array, a.offset, a.ld, b.array, b.offset, b.ld, 1.0f, c, 0, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public MatrixF transAmult(MatrixF B, MatrixF C) {
        return transAmult(1.0f, B, C);
    }

    @Override
    public MatrixF transAmult(float alpha, MatrixF B, MatrixF C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().transAmult(alpha, B, C);
        }
        return transAmultAdd(alpha, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public MatrixF transAmultAdd(MatrixF B, MatrixF C) {
        return transAmultAdd(1.0f, B, C);
    }

    @Override
    public MatrixF transAmultAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkTransAmultAdd(this, B, C);
        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF a = operand(c);
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        Matrices.getBlas().sgemm(TTrans.TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), rows, alpha,
                a.array, a.offset, a.ld, b.array, b.offset, b.ld, 1.0f, c, 0, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public MatrixF transBmult(MatrixF B, MatrixF C) {
        return transBmult(1.0f, B, C);
    }

    @Override
    public MatrixF transBmult(float alpha, MatrixF B, MatrixF C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().transBmult(alpha, B, C);
        }
        return transBmultAdd(alpha, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public MatrixF transBmultAdd(MatrixF B, MatrixF C) {
        return transBmultAdd(1.0f, B, C);
    }

    @Override
    public MatrixF transBmultAdd(float alpha, MatrixF B, MatrixF C) {
        Checks.checkTransBmultAdd(this, B, C);
        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF a = operand(c);
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        Matrices.getBlas().sgemm(TTrans.NO_TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), cols, alpha,
                a.array, a.offset, a.ld, b.array, b.offset, b.ld, 1.0f, c, 0, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public MatrixF times(MatrixF B) {
        return mult(B, Matrices.createF(rows, B.numColumns()));
    }

    @Override
    public MatrixF timesTimes(MatrixF B, MatrixF C) {
        if (Matrices.aTimesBfirst(this, B, C)) {
            return mult(B, Matrices.createF(rows, B.numColumns())).mult(C, Matrices.createF(rows, C.numColumns()));
        }
        return mult(B.mult(C, Matrices.createF(B.numRows(), C.numColumns())), Matrices.createF(rows, C.numColumns()));
    }

    @Override
    public MatrixF timesTransposed() {
        return transBmult(this, Matrices.createF(rows, rows));
    }

    @Override
    public MatrixF timesTransposed(MatrixF B) {
        return transBmult(B, Matrices.createF(rows, B.numRows()));
    }

    @Override
    public MatrixF transposedTimes() {
        return transAmult(this, Matrices.createF(cols, cols));
    }

    @Override
    public MatrixF transposedTimes(MatrixF B) {
        return transAmult(B, Matrices.createF(cols, B.numColumns()));
    }

    @Override
    public MatrixF timesPlus(MatrixF B, MatrixF C) {
        return multAdd(B, C.copy());
    }

    @Override
    public MatrixF timesMinus(MatrixF B, MatrixF C) {
        return multAdd(B, C.uminus());
    }

    // reads on the parent or decompositions that copy their input

    @Override
    public MatrixF submatrix(int row0, int col0, int row1, int col1, MatrixF B, int rb, int cb) {
        // the parent only knows its own, larger bounds
        checkSubmatrixIndexes(row0, col0, row1, col1);
        return parent.submatrix(r0 + row0, c0 + col0, r0 + row1, c0 + col1, B, rb, cb);
    }

    @Override
    public SvdF svd(boolean full) {
        return new SvdF(this, full);
    }

    @Override
    public SvdEconF svdEcon() {
        return new SvdEconF(this);
    }

    @Override
    public float[] singularValues() {
        return svd(false).getS();
    }

    @Override
    public EvdF evd(boolean full) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("EVD only works for square matrices");
        }
        return new EvdF(this, full);
    }

    @Override
    public QrdF qrd() {
        return new QrdF(this);
    }

    @Override
    public LudF lud() {
        return new LudF(this);
    }

    @Override
    public float norm2() {
        return new SvdF(this, false).norm2();
    }
}
