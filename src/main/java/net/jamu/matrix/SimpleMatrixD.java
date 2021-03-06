/*
 * Copyright 2019, 2020 Stefan Zobel
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

import net.dedekind.blas.Blas;
import net.frobenius.TTrans;
import net.frobenius.lapack.PlainLapack;

/**
 * A simple dense matrix implementation of a column-major layout double matrix
 * based on {@code BLAS} and {@code LAPACK} routines.
 */
public class SimpleMatrixD extends MatrixDBase implements MatrixD {

    private static final double BETA = 1.0;

    /**
     * Create a new {@code SimpleMatrixD} of dimension {@code (rows, cols)}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     */
    public SimpleMatrixD(int rows, int cols) {
        this(rows, cols, new double[Checks.checkArrayLength(rows, cols)]);
    }

    /**
     * Create a new {@code SimpleMatrixD} of dimension {@code (rows, cols)} with
     * all matrix elements set to {@code initialValue}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     * @param initialValue
     *            the initial value to set
     */
    public SimpleMatrixD(int rows, int cols, double initialValue) {
        super(rows, cols, new double[Checks.checkArrayLength(rows, cols)], false);
        Arrays.fill(a, initialValue);
    }

    private SimpleMatrixD(SimpleMatrixD other) {
        super(other.rows, other.cols, other.a, true);
    }

    protected SimpleMatrixD(int rows, int cols, double[] data) {
        super(rows, cols, data, false);
    }

    @Override
    protected MatrixD create(int rows, int cols) {
        return new SimpleMatrixD(rows, cols);
    }

    @Override
    protected MatrixD create(int rows, int cols, double[] data) {
        return new SimpleMatrixD(rows, cols, data);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD multAdd(double alpha, MatrixD B, MatrixD C) {
        Checks.checkMultAdd(this, B, C);

        Blas blas = Matrices.getBlas();
        blas.dgemm(TTrans.NO_TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), cols, alpha, a,
                Math.max(1, rows), B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transABmultAdd(double alpha, MatrixD B, MatrixD C) {
        Checks.checkTransABmultAdd(this, B, C);

        Blas blas = Matrices.getBlas();
        blas.dgemm(TTrans.TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), rows, alpha, a,
                Math.max(1, rows), B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transAmultAdd(double alpha, MatrixD B, MatrixD C) {
        Checks.checkTransAmultAdd(this, B, C);

        Blas blas = Matrices.getBlas();
        blas.dgemm(TTrans.TRANS.val(), TTrans.NO_TRANS.val(), C.numRows(), C.numColumns(), rows, alpha, a,
                Math.max(1, rows), B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD transBmultAdd(double alpha, MatrixD B, MatrixD C) {
        Checks.checkTransBmultAdd(this, B, C);

        Blas blas = Matrices.getBlas();
        blas.dgemm(TTrans.NO_TRANS.val(), TTrans.TRANS.val(), C.numRows(), C.numColumns(), cols, alpha, a,
                Math.max(1, rows), B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD solve(MatrixD B, MatrixD X) {
        Checks.checkSolve(this, B, X);
        if (this.isSquareMatrix()) {
            return lusolve(this, X, B);
        }
        return qrsolve(this, X, B);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public SvdD svd(boolean full) {
        return new SvdD(this, full);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public SvdEconD svdEcon() {
        return new SvdEconD(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public EvdD evd(boolean full) {
        if (!this.isSquareMatrix()) {
            throw new IllegalArgumentException("EVD only works for square matrices");
        }
        return new EvdD(this, full);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public QrdD qrd() {
        return new QrdD(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public LudD lud() {
        return new LudD(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double norm2() {
        return new SvdD(this, false).norm2();
    }

    private static MatrixD lusolve(MatrixD A, MatrixD X, MatrixD B) {
        X.setInplace(B);
        PlainLapack.dgesv(Matrices.getLapack(), A.numRows(), B.numColumns(), A.getArrayUnsafe().clone(),
                Math.max(1, A.numRows()), new int[A.numRows()], X.getArrayUnsafe(), Math.max(1, A.numRows()));
        return X;
    }

    private static MatrixD qrsolve(MatrixD A, MatrixD X, MatrixD B) {
        int rhsCount = B.numColumns();
        int mm = A.numRows();
        int nn = A.numColumns();

        SimpleMatrixD tmp = new SimpleMatrixD(Math.max(mm, nn), rhsCount);
        for (int j = 0; j < rhsCount; ++j) {
            for (int i = 0; i < mm; ++i) {
                tmp.setUnsafe(i, j, B.getUnsafe(i, j));
            }
        }

        PlainLapack.dgels(Matrices.getLapack(), TTrans.NO_TRANS, mm, nn, rhsCount, A.getArrayUnsafe().clone(),
                Math.max(1, mm), tmp.getArrayUnsafe(), Math.max(1, Math.max(mm, nn)));

        for (int j = 0; j < rhsCount; ++j) {
            for (int i = 0; i < nn; ++i) {
                X.setUnsafe(i, j, tmp.getUnsafe(i, j));
            }
        }
        return X;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MatrixD copy() {
        return new SimpleMatrixD(this);
    }
}
