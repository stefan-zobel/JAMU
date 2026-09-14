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

import java.util.Objects;

import net.dedekind.blas.BlasExt;
import net.dedekind.blas.Trans;
import net.jamu.complex.Zf;

/**
 * A read-only, live view of a rectangular region of another
 * {@code ComplexMatrixF}.
 */
final class ComplexMatrixFView extends DimensionsBase implements ComplexMatrixF {

    private static final String READ_ONLY = "a matrix view is read-only";

    private final ComplexMatrixF parent;
    private final int r0;
    private final int c0;

    ComplexMatrixFView(ComplexMatrixF parent, int r0, int c0, int rows, int cols) {
        super(rows, cols, true, Float.TYPE);
        this.parent = parent;
        this.r0 = r0;
        this.c0 = c0;
    }

    static ComplexMatrixF create(ComplexMatrixF A, int r0, int c0, int r1, int c1) {
        if (A == null) {
            throw new NullPointerException("A");
        }
        A.checkSubmatrixIndexes(r0, c0, r1, c1);
        if (A instanceof ComplexMatrixFView) {
            ComplexMatrixFView v = (ComplexMatrixFView) A;
            return new ComplexMatrixFView(v.parent, v.r0 + r0, v.c0 + c0, r1 - r0 + 1, c1 - c0 + 1);
        }
        return new ComplexMatrixFView(A, r0, c0, r1 - r0 + 1, c1 - c0 + 1);
    }

    /**
     * Returns whether this view reads from {@code array}.
     */
    boolean shares(float[] array) {
        return parent.getArrayUnsafe() == array;
    }

    /**
     * Returns this view as a cgemm operand at offset 0, read from a copy if its
     * parent is {@code out} or it does not start at {@code (0, 0)}.
     */
    ReadAccess.OperandF operand(float[] out) {
        float[] p = parent.getArrayUnsafe();
        // cgemm has no offset, so only a view at (0, 0) can be read in place
        if (p == out || r0 != 0 || c0 != 0) {
            return new ReadAccess.OperandF(copy().getArrayUnsafe(), 0, rows);
        }
        // the region lies within the parent, so ld >= rows keeps every index in bounds
        return new ReadAccess.OperandF(p, 0, parent.numRows());
    }

    /**
     * Throws if {@code m} is a view.
     *
     * @throws UnsupportedOperationException
     *             if {@code m} is a view
     */
    static void refuse(ComplexMatrixF m) {
        if (m instanceof ComplexMatrixFView) {
            throw readOnly();
        }
    }

    private static UnsupportedOperationException readOnly() {
        return new UnsupportedOperationException(READ_ONLY);
    }

    // direct reads

    @Override
    public void get(int row, int col, Zf out) {
        Objects.requireNonNull(out);
        checkIndex(row, col);
        parent.getUnsafe(r0 + row, c0 + col, out);
    }

    @Override
    public Zf get(int row, int col) {
        checkIndex(row, col);
        return parent.getUnsafe(r0 + row, c0 + col);
    }

    @Override
    public void getUnsafe(int row, int col, Zf out) {
        parent.getUnsafe(r0 + row, c0 + col, out);
    }

    @Override
    public Zf getUnsafe(int row, int col) {
        return parent.getUnsafe(r0 + row, c0 + col);
    }

    @Override
    public Zf toScalar() {
        if (!isScalar()) {
            throw new IllegalStateException("(" + rows + " x " + cols + ") matrix is not a scalar");
        }
        return parent.getUnsafe(r0, c0);
    }

    @Override
    public ComplexMatrixF copy() {
        ComplexMatrixF C = Matrices.createComplexF(rows, cols);
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
    public ComplexMatrixF scaleInplace(float alphar, float alphai) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF addInplace(ComplexMatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF addInplace(float alphar, float alphai, ComplexMatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF zeroInplace() {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF setInplace(ComplexMatrixF other) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF setInplace(float alphar, float alphai, ComplexMatrixF other) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF setColumnInplace(int colIdx, ComplexMatrixF colVector) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF setInplaceUpperTrapezoidal(ComplexMatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF setInplaceLowerTrapezoidal(ComplexMatrixF B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF set(int row, int col, float valr, float vali) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF add(int row, int col, float valr, float vali) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF setSubmatrixInplace(int r0, int c0, ComplexMatrixF B, int rb0, int cb0, int rb1, int cb1) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF zeroizeSubEpsilonInplace(int k) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF zeroizeSubEpsilonRelativeInplace(int k) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF sanitizeNonFiniteInplace(float nanSurrogate, float posInfSurrogate,
            float negInfSurrogate) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixF sanitizeNaNInplace(float nanSurrogate) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public void setUnsafe(int row, int col, float valr, float vali) {
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
    public ComplexMatrixF scale(float alphar, float alphai, ComplexMatrixF B) {
        return copy().scale(alphar, alphai, B);
    }

    @Override
    public ComplexMatrixF conjTrans(ComplexMatrixF AH) {
        Checks.checkTrans(this, AH);
        float[] p = parent.getArrayUnsafe();
        float[] o = AH.getArrayUnsafe();
        // writing into the parent would overwrite what is still to be read
        if (o == p) {
            return copy().conjTrans(AH);
        }
        int ld = parent.numRows();
        for (int col = 0; col < cols; ++col) {
            int off = (c0 + col) * ld + r0;
            for (int row = 0; row < rows; ++row) {
                int pidx = 2 * (off + row);
                int oidx = 2 * (row * cols + col);
                o[oidx] = p[pidx];
                o[oidx + 1] = -p[pidx + 1];
            }
        }
        return AH;
    }

    @Override
    public ComplexMatrixF trans(ComplexMatrixF AT) {
        Checks.checkTrans(this, AT);
        float[] p = parent.getArrayUnsafe();
        float[] o = AT.getArrayUnsafe();
        // writing into the parent would overwrite what is still to be read
        if (o == p) {
            return copy().trans(AT);
        }
        int ld = parent.numRows();
        for (int col = 0; col < cols; ++col) {
            int off = (c0 + col) * ld + r0;
            for (int row = 0; row < rows; ++row) {
                int pidx = 2 * (off + row);
                int oidx = 2 * (row * cols + col);
                o[oidx] = p[pidx];
                o[oidx + 1] = p[pidx + 1];
            }
        }
        return AT;
    }

    @Override
    public ComplexMatrixF add(ComplexMatrixF B, ComplexMatrixF C) {
        return copy().add(B, C);
    }

    @Override
    public ComplexMatrixF add(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        return copy().add(alphar, alphai, B, C);
    }

    @Override
    public ComplexMatrixF solve(ComplexMatrixF B, ComplexMatrixF X) {
        Checks.checkSolve(this, B, X);
        // the copy is the LAPACK work array, so the solver must not copy again
        float[] work = copy().getArrayUnsafe();
        if (isSquareMatrix()) {
            return SimpleComplexMatrixF.lusolve(work, rows, X, B);
        }
        return SimpleComplexMatrixF.qrsolve(work, rows, cols, X, B);
    }

    @Override
    public ComplexMatrixF inv(ComplexMatrixF inverse) {
        if (!isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        Checks.checkEqualDimension(this, inverse);
        return solve(Matrices.identityComplexF(rows), inverse);
    }

    @Override
    public ComplexMatrixF pseudoInv() {
        return ComplexMatrixFBase.pseudoInv(svd(true), rows, cols);
    }

    @Override
    public ComplexMatrixF expm() {
        return copy().expm();
    }

    @Override
    public ComplexMatrixF hadamard(ComplexMatrixF B, ComplexMatrixF out) {
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
    public Zf trace() {
        return copy().trace();
    }

    @Override
    public ComplexMatrixF selectColumn(int col) {
        return copy().selectColumn(col);
    }

    @Override
    public ComplexMatrixF selectConsecutiveColumns(int colFrom, int colTo) {
        return copy().selectConsecutiveColumns(colFrom, colTo);
    }

    @Override
    public ComplexMatrixF selectSubmatrix(int rowFrom, int colFrom, int rowTo, int colTo) {
        return copy().selectSubmatrix(rowFrom, colFrom, rowTo, colTo);
    }

    @Override
    public ComplexMatrixF appendColumn(ComplexMatrixF colVector) {
        return copy().appendColumn(colVector);
    }

    @Override
    public ComplexMatrixF appendMatrix(ComplexMatrixF matrix) {
        return copy().appendMatrix(matrix);
    }

    @Override
    public ComplexMatrixF mldivide(ComplexMatrixF B) {
        Checks.checkSameRows(this, B);
        return solve(B, Matrices.createComplexF(cols, B.numColumns()));
    }

    @Override
    public ComplexMatrixF mrdivide(ComplexMatrixF B) {
        return SimpleComplexMatrixF.mrdivide(this, B);
    }

    @Override
    public ComplexMatrixF timesMany(ComplexMatrixF m, ComplexMatrixF... matrices) {
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
    public ComplexMatrixF plus(ComplexMatrixF B) {
        return copy().plus(B);
    }

    @Override
    public ComplexMatrixF minus(ComplexMatrixF B) {
        return copy().minus(B);
    }

    @Override
    public ComplexMatrixF uminus() {
        return copy().uminus();
    }

    @Override
    public ComplexMatrixF abs() {
        return copy().abs();
    }

    @Override
    public ComplexMatrixF conjugateTranspose() {
        return conjTrans(Matrices.createComplexF(cols, rows));
    }

    @Override
    public ComplexMatrixF transpose() {
        return trans(Matrices.createComplexF(cols, rows));
    }

    @Override
    public ComplexMatrixF inverse() {
        if (!isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        return inv(Matrices.createComplexF(rows, cols));
    }

    @Override
    public ComplexMatrixF hadamard(ComplexMatrixF B) {
        return copy().hadamard(B);
    }

    @Override
    public ComplexMatrixF reshape(int rows, int cols) {
        return copy().reshape(rows, cols);
    }

    @Override
    public MatrixF toRealMatrix() {
        return copy().toRealMatrix();
    }

    // products through cgemm on the parent

    @Override
    public ComplexMatrixF mult(ComplexMatrixF B, ComplexMatrixF C) {
        return mult(1.0f, 0.0f, B, C);
    }

    @Override
    public ComplexMatrixF mult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        if (shares(C.getArrayUnsafe())) {
            // zeroing the output would zero this view
            return copy().mult(alphar, alphai, B, C);
        }
        return multAdd(alphar, alphai, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public ComplexMatrixF multAdd(ComplexMatrixF B, ComplexMatrixF C) {
        return multAdd(1.0f, 0.0f, B, C);
    }

    @Override
    public ComplexMatrixF multAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        Checks.checkMultAdd(this, B, C);
        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF a = operand(c);
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        BlasExt.getInstance().cgemm3m(Trans.N, Trans.N, C.numRows(), C.numColumns(), cols, alphar, alphai, a.array,
                a.ld, b.array, b.ld, 1.0f, 0.0f, c, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public ComplexMatrixF conjTransABmult(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransABmult(1.0f, 0.0f, B, C);
    }

    @Override
    public ComplexMatrixF conjTransABmult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().conjTransABmult(alphar, alphai, B, C);
        }
        return conjTransABmultAdd(alphar, alphai, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public ComplexMatrixF conjTransABmultAdd(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransABmultAdd(1.0f, 0.0f, B, C);
    }

    @Override
    public ComplexMatrixF conjTransABmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        Checks.checkTransABmultAdd(this, B, C);
        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF a = operand(c);
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        BlasExt.getInstance().cgemm3m(Trans.C, Trans.C, C.numRows(), C.numColumns(), rows, alphar, alphai, a.array,
                a.ld, b.array, b.ld, 1.0f, 0.0f, c, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public ComplexMatrixF conjTransAmult(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransAmult(1.0f, 0.0f, B, C);
    }

    @Override
    public ComplexMatrixF conjTransAmult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().conjTransAmult(alphar, alphai, B, C);
        }
        return conjTransAmultAdd(alphar, alphai, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public ComplexMatrixF conjTransAmultAdd(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransAmultAdd(1.0f, 0.0f, B, C);
    }

    @Override
    public ComplexMatrixF conjTransAmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        Checks.checkTransAmultAdd(this, B, C);
        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF a = operand(c);
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        BlasExt.getInstance().cgemm3m(Trans.C, Trans.N, C.numRows(), C.numColumns(), rows, alphar, alphai, a.array,
                a.ld, b.array, b.ld, 1.0f, 0.0f, c, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public ComplexMatrixF conjTransBmult(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransBmult(1.0f, 0.0f, B, C);
    }

    @Override
    public ComplexMatrixF conjTransBmult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().conjTransBmult(alphar, alphai, B, C);
        }
        return conjTransBmultAdd(alphar, alphai, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public ComplexMatrixF conjTransBmultAdd(ComplexMatrixF B, ComplexMatrixF C) {
        return conjTransBmultAdd(1.0f, 0.0f, B, C);
    }

    @Override
    public ComplexMatrixF conjTransBmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        Checks.checkTransBmultAdd(this, B, C);
        float[] c = C.getArrayUnsafe();
        ReadAccess.OperandF a = operand(c);
        ReadAccess.OperandF b = ReadAccess.operand(B, c);
        BlasExt.getInstance().cgemm3m(Trans.N, Trans.C, C.numRows(), C.numColumns(), cols, alphar, alphai, a.array,
                a.ld, b.array, b.ld, 1.0f, 0.0f, c, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public ComplexMatrixF times(ComplexMatrixF B) {
        return mult(B, Matrices.createComplexF(rows, B.numColumns()));
    }

    @Override
    public ComplexMatrixF timesTimes(ComplexMatrixF B, ComplexMatrixF C) {
        if (Matrices.aTimesBfirst(this, B, C)) {
            return mult(B, Matrices.createComplexF(rows, B.numColumns())).mult(C,
                    Matrices.createComplexF(rows, C.numColumns()));
        }
        return mult(B.mult(C, Matrices.createComplexF(B.numRows(), C.numColumns())),
                Matrices.createComplexF(rows, C.numColumns()));
    }

    @Override
    public ComplexMatrixF timesConjugateTransposed() {
        return conjTransBmult(this, Matrices.createComplexF(rows, rows));
    }

    @Override
    public ComplexMatrixF timesConjugateTransposed(ComplexMatrixF B) {
        return conjTransBmult(B, Matrices.createComplexF(rows, B.numRows()));
    }

    @Override
    public ComplexMatrixF conjugateTransposedTimes() {
        return conjTransAmult(this, Matrices.createComplexF(cols, cols));
    }

    @Override
    public ComplexMatrixF conjugateTransposedTimes(ComplexMatrixF B) {
        return conjTransAmult(B, Matrices.createComplexF(cols, B.numColumns()));
    }

    @Override
    public ComplexMatrixF times(MatrixF B) {
        Checks.checkMult(this, B);
        return times(Matrices.convertToComplex(B));
    }

    @Override
    public ComplexMatrixF timesPlus(ComplexMatrixF B, ComplexMatrixF C) {
        return multAdd(B, C.copy());
    }

    @Override
    public ComplexMatrixF timesMinus(ComplexMatrixF B, ComplexMatrixF C) {
        return multAdd(B, C.uminus());
    }

    // reads on the parent or decompositions that copy their input

    @Override
    public ComplexMatrixF submatrix(int row0, int col0, int row1, int col1, ComplexMatrixF B, int rb, int cb) {
        // the parent only knows its own, larger bounds
        checkSubmatrixIndexes(row0, col0, row1, col1);
        return parent.submatrix(r0 + row0, c0 + col0, r0 + row1, c0 + col1, B, rb, cb);
    }

    @Override
    public SvdComplexF svd(boolean full) {
        return new SvdComplexF(this, full);
    }

    @Override
    public SvdEconComplexF svdEcon() {
        return new SvdEconComplexF(this);
    }

    @Override
    public float[] singularValues() {
        return svd(false).getS();
    }

    @Override
    public EvdComplexF evd(boolean full) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("EVD only works for square matrices");
        }
        // unlike the other decompositions, EvdComplexF reads the array of its input
        return new EvdComplexF(copy(), full);
    }

    @Override
    public QrdComplexF qrd() {
        return new QrdComplexF(this);
    }

    @Override
    public LudComplexF lud() {
        return new LudComplexF(this);
    }

    @Override
    public float norm2() {
        return new SvdComplexF(this, false).norm2();
    }
}
