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
import net.jamu.complex.Zd;

/**
 * A read-only, live view of a rectangular region of another
 * {@code ComplexMatrixD}.
 */
final class ComplexMatrixDView extends DimensionsBase implements ComplexMatrixD {

    private static final String READ_ONLY = "a matrix view is read-only";

    private final ComplexMatrixD parent;
    private final int r0;
    private final int c0;

    ComplexMatrixDView(ComplexMatrixD parent, int r0, int c0, int rows, int cols) {
        super(rows, cols, true, Double.TYPE);
        this.parent = parent;
        this.r0 = r0;
        this.c0 = c0;
    }

    static ComplexMatrixD create(ComplexMatrixD A, int r0, int c0, int r1, int c1) {
        if (A == null) {
            throw new NullPointerException("A");
        }
        A.checkSubmatrixIndexes(r0, c0, r1, c1);
        if (A instanceof ComplexMatrixDView) {
            ComplexMatrixDView v = (ComplexMatrixDView) A;
            return new ComplexMatrixDView(v.parent, v.r0 + r0, v.c0 + c0, r1 - r0 + 1, c1 - c0 + 1);
        }
        return new ComplexMatrixDView(A, r0, c0, r1 - r0 + 1, c1 - c0 + 1);
    }

    /**
     * Returns whether this view reads from {@code array}.
     */
    boolean shares(double[] array) {
        return parent.getArrayUnsafe() == array;
    }

    /**
     * Returns this view as a zgemm operand at offset 0, read from a copy if its
     * parent is {@code out} or it does not start at {@code (0, 0)}.
     */
    ReadAccess.OperandD operand(double[] out) {
        double[] p = parent.getArrayUnsafe();
        // zgemm has no offset, so only a view at (0, 0) can be read in place
        if (p == out || r0 != 0 || c0 != 0) {
            return new ReadAccess.OperandD(copy().getArrayUnsafe(), 0, rows);
        }
        // the region lies within the parent, so ld >= rows keeps every index in bounds
        return new ReadAccess.OperandD(p, 0, parent.numRows());
    }

    /**
     * Throws if {@code m} is a view.
     *
     * @throws UnsupportedOperationException
     *             if {@code m} is a view
     */
    static void refuse(ComplexMatrixD m) {
        if (m instanceof ComplexMatrixDView) {
            throw readOnly();
        }
    }

    private static UnsupportedOperationException readOnly() {
        return new UnsupportedOperationException(READ_ONLY);
    }

    // direct reads

    @Override
    public void get(int row, int col, Zd out) {
        Objects.requireNonNull(out);
        checkIndex(row, col);
        parent.getUnsafe(r0 + row, c0 + col, out);
    }

    @Override
    public Zd get(int row, int col) {
        checkIndex(row, col);
        return parent.getUnsafe(r0 + row, c0 + col);
    }

    @Override
    public void getUnsafe(int row, int col, Zd out) {
        parent.getUnsafe(r0 + row, c0 + col, out);
    }

    @Override
    public Zd getUnsafe(int row, int col) {
        return parent.getUnsafe(r0 + row, c0 + col);
    }

    @Override
    public Zd toScalar() {
        if (!isScalar()) {
            throw new IllegalStateException("(" + rows + " x " + cols + ") matrix is not a scalar");
        }
        return parent.getUnsafe(r0, c0);
    }

    @Override
    public ComplexMatrixD copy() {
        ComplexMatrixD C = Matrices.createComplexD(rows, cols);
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
    public ComplexMatrixD scaleInplace(double alphar, double alphai) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD addInplace(ComplexMatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD addInplace(double alphar, double alphai, ComplexMatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD zeroInplace() {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD setInplace(ComplexMatrixD other) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD setInplace(double alphar, double alphai, ComplexMatrixD other) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD setColumnInplace(int colIdx, ComplexMatrixD colVector) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD setInplaceUpperTrapezoidal(ComplexMatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD setInplaceLowerTrapezoidal(ComplexMatrixD B) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD set(int row, int col, double valr, double vali) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD add(int row, int col, double valr, double vali) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD setSubmatrixInplace(int r0, int c0, ComplexMatrixD B, int rb0, int cb0, int rb1, int cb1) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD zeroizeSubEpsilonInplace(int k) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD zeroizeSubEpsilonRelativeInplace(int k) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD sanitizeNonFiniteInplace(double nanSurrogate, double posInfSurrogate,
            double negInfSurrogate) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public ComplexMatrixD sanitizeNaNInplace(double nanSurrogate) {
        throw readOnly();
    }

    /**
     * Not supported by a read-only view.
     *
     * @throws UnsupportedOperationException
     *             always
     */
    @Override
    public void setUnsafe(int row, int col, double valr, double vali) {
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
    public ComplexMatrixD scale(double alphar, double alphai, ComplexMatrixD B) {
        return copy().scale(alphar, alphai, B);
    }

    @Override
    public ComplexMatrixD conjTrans(ComplexMatrixD AH) {
        Checks.checkTrans(this, AH);
        double[] p = parent.getArrayUnsafe();
        double[] o = AH.getArrayUnsafe();
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
    public ComplexMatrixD trans(ComplexMatrixD AT) {
        Checks.checkTrans(this, AT);
        double[] p = parent.getArrayUnsafe();
        double[] o = AT.getArrayUnsafe();
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
    public ComplexMatrixD add(ComplexMatrixD B, ComplexMatrixD C) {
        return copy().add(B, C);
    }

    @Override
    public ComplexMatrixD add(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        return copy().add(alphar, alphai, B, C);
    }

    @Override
    public ComplexMatrixD solve(ComplexMatrixD B, ComplexMatrixD X) {
        Checks.checkSolve(this, B, X);
        // the copy is the LAPACK work array, so the solver must not copy again
        double[] work = copy().getArrayUnsafe();
        if (isSquareMatrix()) {
            return SimpleComplexMatrixD.lusolve(work, rows, X, B);
        }
        return SimpleComplexMatrixD.qrsolve(work, rows, cols, X, B);
    }

    @Override
    public ComplexMatrixD inv(ComplexMatrixD inverse) {
        if (!isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        Checks.checkEqualDimension(this, inverse);
        return solve(Matrices.identityComplexD(rows), inverse);
    }

    @Override
    public ComplexMatrixD pseudoInv() {
        return ComplexMatrixDBase.pseudoInv(svd(true), rows, cols);
    }

    @Override
    public ComplexMatrixD expm() {
        return copy().expm();
    }

    @Override
    public ComplexMatrixD hadamard(ComplexMatrixD B, ComplexMatrixD out) {
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
    public Zd trace() {
        return copy().trace();
    }

    @Override
    public ComplexMatrixD selectColumn(int col) {
        return copy().selectColumn(col);
    }

    @Override
    public ComplexMatrixD selectConsecutiveColumns(int colFrom, int colTo) {
        return copy().selectConsecutiveColumns(colFrom, colTo);
    }

    @Override
    public ComplexMatrixD selectSubmatrix(int rowFrom, int colFrom, int rowTo, int colTo) {
        return copy().selectSubmatrix(rowFrom, colFrom, rowTo, colTo);
    }

    @Override
    public ComplexMatrixD appendColumn(ComplexMatrixD colVector) {
        return copy().appendColumn(colVector);
    }

    @Override
    public ComplexMatrixD appendMatrix(ComplexMatrixD matrix) {
        return copy().appendMatrix(matrix);
    }

    @Override
    public ComplexMatrixD mldivide(ComplexMatrixD B) {
        Checks.checkSameRows(this, B);
        return solve(B, Matrices.createComplexD(cols, B.numColumns()));
    }

    @Override
    public ComplexMatrixD mrdivide(ComplexMatrixD B) {
        return SimpleComplexMatrixD.mrdivide(this, B);
    }

    @Override
    public ComplexMatrixD timesMany(ComplexMatrixD m, ComplexMatrixD... matrices) {
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
    public ComplexMatrixD plus(ComplexMatrixD B) {
        return copy().plus(B);
    }

    @Override
    public ComplexMatrixD minus(ComplexMatrixD B) {
        return copy().minus(B);
    }

    @Override
    public ComplexMatrixD uminus() {
        return copy().uminus();
    }

    @Override
    public ComplexMatrixD abs() {
        return copy().abs();
    }

    @Override
    public ComplexMatrixD conjugateTranspose() {
        return conjTrans(Matrices.createComplexD(cols, rows));
    }

    @Override
    public ComplexMatrixD transpose() {
        return trans(Matrices.createComplexD(cols, rows));
    }

    @Override
    public ComplexMatrixD inverse() {
        if (!isSquareMatrix()) {
            throw new IllegalArgumentException("The inverse is only defined for square matrices");
        }
        return inv(Matrices.createComplexD(rows, cols));
    }

    @Override
    public ComplexMatrixD hadamard(ComplexMatrixD B) {
        return copy().hadamard(B);
    }

    @Override
    public ComplexMatrixD reshape(int rows, int cols) {
        return copy().reshape(rows, cols);
    }

    @Override
    public MatrixD toRealMatrix() {
        return copy().toRealMatrix();
    }

    // products through zgemm on the parent

    @Override
    public ComplexMatrixD mult(ComplexMatrixD B, ComplexMatrixD C) {
        return mult(1.0, 0.0, B, C);
    }

    @Override
    public ComplexMatrixD mult(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        if (shares(C.getArrayUnsafe())) {
            // zeroing the output would zero this view
            return copy().mult(alphar, alphai, B, C);
        }
        return multAdd(alphar, alphai, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public ComplexMatrixD multAdd(ComplexMatrixD B, ComplexMatrixD C) {
        return multAdd(1.0, 0.0, B, C);
    }

    @Override
    public ComplexMatrixD multAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkMultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD a = operand(c);
        ReadAccess.OperandD b = ReadAccess.operand(B, c);
        BlasExt.getInstance().zgemm3m(Trans.N, Trans.N, C.numRows(), C.numColumns(), cols, alphar, alphai, a.array,
                a.ld, b.array, b.ld, 1.0, 0.0, c, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public ComplexMatrixD conjTransABmult(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransABmult(1.0, 0.0, B, C);
    }

    @Override
    public ComplexMatrixD conjTransABmult(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().conjTransABmult(alphar, alphai, B, C);
        }
        return conjTransABmultAdd(alphar, alphai, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public ComplexMatrixD conjTransABmultAdd(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransABmultAdd(1.0, 0.0, B, C);
    }

    @Override
    public ComplexMatrixD conjTransABmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkTransABmultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD a = operand(c);
        ReadAccess.OperandD b = ReadAccess.operand(B, c);
        BlasExt.getInstance().zgemm3m(Trans.C, Trans.C, C.numRows(), C.numColumns(), rows, alphar, alphai, a.array,
                a.ld, b.array, b.ld, 1.0, 0.0, c, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public ComplexMatrixD conjTransAmult(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransAmult(1.0, 0.0, B, C);
    }

    @Override
    public ComplexMatrixD conjTransAmult(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().conjTransAmult(alphar, alphai, B, C);
        }
        return conjTransAmultAdd(alphar, alphai, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public ComplexMatrixD conjTransAmultAdd(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransAmultAdd(1.0, 0.0, B, C);
    }

    @Override
    public ComplexMatrixD conjTransAmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkTransAmultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD a = operand(c);
        ReadAccess.OperandD b = ReadAccess.operand(B, c);
        BlasExt.getInstance().zgemm3m(Trans.C, Trans.N, C.numRows(), C.numColumns(), rows, alphar, alphai, a.array,
                a.ld, b.array, b.ld, 1.0, 0.0, c, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public ComplexMatrixD conjTransBmult(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransBmult(1.0, 0.0, B, C);
    }

    @Override
    public ComplexMatrixD conjTransBmult(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        if (shares(C.getArrayUnsafe())) {
            return copy().conjTransBmult(alphar, alphai, B, C);
        }
        return conjTransBmultAdd(alphar, alphai, ReadAccess.detach(B, C), C.zeroInplace());
    }

    @Override
    public ComplexMatrixD conjTransBmultAdd(ComplexMatrixD B, ComplexMatrixD C) {
        return conjTransBmultAdd(1.0, 0.0, B, C);
    }

    @Override
    public ComplexMatrixD conjTransBmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkTransBmultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD a = operand(c);
        ReadAccess.OperandD b = ReadAccess.operand(B, c);
        BlasExt.getInstance().zgemm3m(Trans.N, Trans.C, C.numRows(), C.numColumns(), cols, alphar, alphai, a.array,
                a.ld, b.array, b.ld, 1.0, 0.0, c, Math.max(1, C.numRows()));
        return C;
    }

    @Override
    public ComplexMatrixD times(ComplexMatrixD B) {
        return mult(B, Matrices.createComplexD(rows, B.numColumns()));
    }

    @Override
    public ComplexMatrixD timesTimes(ComplexMatrixD B, ComplexMatrixD C) {
        if (Matrices.aTimesBfirst(this, B, C)) {
            return mult(B, Matrices.createComplexD(rows, B.numColumns())).mult(C,
                    Matrices.createComplexD(rows, C.numColumns()));
        }
        return mult(B.mult(C, Matrices.createComplexD(B.numRows(), C.numColumns())),
                Matrices.createComplexD(rows, C.numColumns()));
    }

    @Override
    public ComplexMatrixD timesConjugateTransposed() {
        return conjTransBmult(this, Matrices.createComplexD(rows, rows));
    }

    @Override
    public ComplexMatrixD timesConjugateTransposed(ComplexMatrixD B) {
        return conjTransBmult(B, Matrices.createComplexD(rows, B.numRows()));
    }

    @Override
    public ComplexMatrixD conjugateTransposedTimes() {
        return conjTransAmult(this, Matrices.createComplexD(cols, cols));
    }

    @Override
    public ComplexMatrixD conjugateTransposedTimes(ComplexMatrixD B) {
        return conjTransAmult(B, Matrices.createComplexD(cols, B.numColumns()));
    }

    @Override
    public ComplexMatrixD times(MatrixD B) {
        Checks.checkMult(this, B);
        return times(Matrices.convertToComplex(B));
    }

    @Override
    public ComplexMatrixD timesPlus(ComplexMatrixD B, ComplexMatrixD C) {
        return multAdd(B, C.copy());
    }

    @Override
    public ComplexMatrixD timesMinus(ComplexMatrixD B, ComplexMatrixD C) {
        return multAdd(B, C.uminus());
    }

    // reads on the parent or decompositions that copy their input

    @Override
    public ComplexMatrixD submatrix(int row0, int col0, int row1, int col1, ComplexMatrixD B, int rb, int cb) {
        // the parent only knows its own, larger bounds
        checkSubmatrixIndexes(row0, col0, row1, col1);
        return parent.submatrix(r0 + row0, c0 + col0, r0 + row1, c0 + col1, B, rb, cb);
    }

    @Override
    public SvdComplexD svd(boolean full) {
        return new SvdComplexD(this, full);
    }

    @Override
    public SvdEconComplexD svdEcon() {
        return new SvdEconComplexD(this);
    }

    @Override
    public double[] singularValues() {
        return svd(false).getS();
    }

    @Override
    public EvdComplexD evd(boolean full) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("EVD only works for square matrices");
        }
        // unlike the other decompositions, EvdComplexD reads the array of its input
        return new EvdComplexD(copy(), full);
    }

    @Override
    public QrdComplexD qrd() {
        return new QrdComplexD(this);
    }

    @Override
    public LudComplexD lud() {
        return new LudComplexD(this);
    }

    @Override
    public double norm2() {
        return new SvdComplexD(this, false).norm2();
    }
}
