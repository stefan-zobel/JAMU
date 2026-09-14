/*
 * Copyright 2020 Stefan Zobel
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

import java.util.Arrays;

import net.dedekind.blas.BlasExt;
import net.dedekind.blas.Trans;
import net.dedekind.lapack.Lapack;
import net.frobenius.TTrans;
import net.frobenius.lapack.PlainLapack;

/**
 * A simple dense matrix implementation of a column-major layout double
 * precision complex matrix based on {@code BLAS} and {@code LAPACK} routines.
 */
public class SimpleComplexMatrixD extends ComplexMatrixDBase implements ComplexMatrixD {

    private static final double BETA_R = 1.0;
    private static final double BETA_I = 0.0;

    /**
     * Create a new {@code SimpleComplexMatrixD} of dimension
     * {@code (rows, cols)}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     */
    public SimpleComplexMatrixD(int rows, int cols) {
        this(rows, cols, new double[Checks.checkComplexArrayLength(rows, cols)]);
    }

    /**
     * Create a new {@code SimpleComplexMatrixD} of dimension
     * {@code (rows, cols)} with all matrix elements set to
     * {@code initialValue}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param initialValue
     *            the initial value to set
     */
    public SimpleComplexMatrixD(int rows, int cols, double initialValue) {
        super(rows, cols, new double[Checks.checkComplexArrayLength(rows, cols)], false);
        Arrays.fill(a, initialValue);
    }

    /**
     * Create a new {@code SimpleComplexMatrixD} of dimension
     * {@code (rows, cols)} with all matrix elements set to
     * {@code (iniValr, iniVali)}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param iniValr
     *            the real part of the initial value to set
     * @param iniVali
     *            the imaginary part of the initial value to set
     */
    public SimpleComplexMatrixD(int rows, int cols, double iniValr, double iniVali) {
        super(rows, cols, new double[Checks.checkComplexArrayLength(rows, cols)], false);
        double[] a_ = a;
        for (int i = 0; i < a_.length; i += 2) {
            a_[i] = iniValr;
            a_[i + 1] = iniVali;
        }
    }

    private SimpleComplexMatrixD(SimpleComplexMatrixD other) {
        super(other.rows, other.cols, other.a, true);
    }

    protected SimpleComplexMatrixD(int rows, int cols, double[] data) {
        super(rows, cols, data, false);
    }

    @Override
    protected ComplexMatrixD create(int rows, int cols) {
        return new SimpleComplexMatrixD(rows, cols);
    }

    @Override
    protected ComplexMatrixD create(int rows, int cols, double[] data) {
        return new SimpleComplexMatrixD(rows, cols, data);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD multAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkMultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD b = ReadAccess.operand(B, c);

        BlasExt blas = BlasExt.getInstance();
        blas.zgemm3m(Trans.N, Trans.N, C.numRows(), C.numColumns(), cols, alphar, alphai, a, Math.max(1, rows),
                b.array, b.ld, BETA_R, BETA_I, c, Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransABmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkTransABmultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD b = ReadAccess.operand(B, c);

        BlasExt blas = BlasExt.getInstance();
        blas.zgemm3m(Trans.C, Trans.C, C.numRows(), C.numColumns(), rows, alphar, alphai, a, Math.max(1, rows),
                b.array, b.ld, BETA_R, BETA_I, c, Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransAmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkTransAmultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD b = ReadAccess.operand(B, c);

        BlasExt blas = BlasExt.getInstance();
        blas.zgemm3m(Trans.C, Trans.N, C.numRows(), C.numColumns(), rows, alphar, alphai, a, Math.max(1, rows),
                b.array, b.ld, BETA_R, BETA_I, c, Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD conjTransBmultAdd(double alphar, double alphai, ComplexMatrixD B, ComplexMatrixD C) {
        Checks.checkTransBmultAdd(this, B, C);
        double[] c = C.getArrayUnsafe();
        ReadAccess.OperandD b = ReadAccess.operand(B, c);

        BlasExt blas = BlasExt.getInstance();
        blas.zgemm3m(Trans.N, Trans.C, C.numRows(), C.numColumns(), cols, alphar, alphai, a, Math.max(1, rows),
                b.array, b.ld, BETA_R, BETA_I, c, Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD solve(ComplexMatrixD B, ComplexMatrixD X) {
        Checks.checkSolve(this, B, X);
        // clone before X gets written, X may be this matrix
        if (this.isSquareMatrix()) {
            return lusolve(a.clone(), rows, X, B);
        }
        return qrsolve(a.clone(), rows, cols, X, B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public SvdComplexD svd(boolean full) {
        return new SvdComplexD(this, full);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public SvdEconComplexD svdEcon() {
        return new SvdEconComplexD(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public EvdComplexD evd(boolean full) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("EVD only works for square matrices");
        }
        return new EvdComplexD(this, full);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public QrdComplexD qrd() {
        return new QrdComplexD(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public LudComplexD lud() {
        return new LudComplexD(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double norm2() {
        return new SvdComplexD(this, false).norm2();
    }

    // A / B = (B^H \ A^H)^H; both transposes are fresh, so LAPACK may overwrite them
    static ComplexMatrixD mrdivide(ComplexMatrixD A, ComplexMatrixD B) {
        Checks.checkSameCols(A, B);
        ComplexMatrixD BH = B.conjugateTranspose();
        ComplexMatrixD AH = A.conjugateTranspose();
        if (BH.isSquareMatrix()) {
            return lusolve(BH.getArrayUnsafe(), BH.numRows(), AH, AH).conjugateTranspose();
        }
        return qrsolve(BH.getArrayUnsafe(), BH.numRows(), BH.numColumns(),
                Matrices.createComplexD(BH.numColumns(), AH.numColumns()), AH).conjugateTranspose();
    }

    // work holds a private copy of the n x n matrix and gets overwritten
    static ComplexMatrixD lusolve(double[] work, int n, ComplexMatrixD X, ComplexMatrixD B) {
        // X may already hold the right-hand sides
        if (X != B) {
            X.setInplace(B);
        }
        PlainLapack.zgesv(Lapack.getInstance(), n, B.numColumns(), work, Math.max(1, n), new int[n],
                X.getArrayUnsafe(), Math.max(1, n));
        return X;
    }

    // work holds a private copy of the mm x nn matrix and gets overwritten
    static ComplexMatrixD qrsolve(double[] work, int mm, int nn, ComplexMatrixD X, ComplexMatrixD B) {
        int rhsCount = B.numColumns();

        SimpleComplexMatrixD tmp = new SimpleComplexMatrixD(Math.max(mm, nn), rhsCount);
        B.submatrix(0, 0, mm - 1, rhsCount - 1, tmp, 0, 0);

        PlainLapack.zgels(Lapack.getInstance(), TTrans.NO_TRANS, mm, nn, rhsCount, work, Math.max(1, mm),
                tmp.getArrayUnsafe(), Math.max(1, Math.max(mm, nn)));

        return tmp.submatrix(0, 0, nn - 1, rhsCount - 1, X, 0, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixD copy() {
        return new SimpleComplexMatrixD(this);
    }
}
