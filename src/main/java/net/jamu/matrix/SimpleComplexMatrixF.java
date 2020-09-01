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

/**
 * A simple dense matrix implementation of a column-major layout single
 * precision complex matrix based on {@code BLAS} and {@code LAPACK} routines.
 */
public class SimpleComplexMatrixF extends ComplexMatrixFBase implements ComplexMatrixF {

    private static final float BETA_R = 1.0f;
    private static final float BETA_I = 0.0f;

    /**
     * Create a new {@code SimpleComplexMatrixF} of dimension
     * {@code (rows, cols)}.
     * 
     * @param rows
     *            number of matrix rows
     * @param cols
     *            number of matrix columns
     */
    public SimpleComplexMatrixF(int rows, int cols) {
        this(rows, cols, new float[Checks.checkComplexArrayLength(rows, cols)]);
    }

    /**
     * Create a new {@code SimpleComplexMatrixF} of dimension
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
    public SimpleComplexMatrixF(int rows, int cols, float initialValue) {
        super(rows, cols, new float[Checks.checkComplexArrayLength(rows, cols)], false);
        Arrays.fill(a, initialValue);
    }

    /**
     * Create a new {@code SimpleComplexMatrixF} of dimension
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
    public SimpleComplexMatrixF(int rows, int cols, float iniValr, float iniVali) {
        super(rows, cols, new float[Checks.checkComplexArrayLength(rows, cols)], false);
        float[] a_ = a;
        for (int i = 0; i < a_.length; i += 2) {
            a_[i] = iniValr;
            a_[i + 1] = iniVali;
        }
    }

    private SimpleComplexMatrixF(SimpleComplexMatrixF other) {
        super(other.rows, other.cols, other.a, true);
    }

    protected SimpleComplexMatrixF(int rows, int cols, float[] data) {
        super(rows, cols, data, false);
    }

    @Override
    protected ComplexMatrixF create(int rows, int cols) {
        return new SimpleComplexMatrixF(rows, cols);
    }

    @Override
    protected ComplexMatrixF create(int rows, int cols, float[] data) {
        return new SimpleComplexMatrixF(rows, cols, data);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF multAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        Checks.checkMultAdd(this, B, C);

        BlasExt blas = BlasExt.getInstance();
        blas.cgemm3m(Trans.N, Trans.N, C.numRows(), C.numColumns(), cols, alphar, alphai, a, Math.max(1, rows),
                B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA_R, BETA_I, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransABmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        Checks.checkTransABmultAdd(this, B, C);

        BlasExt blas = BlasExt.getInstance();
        blas.cgemm3m(Trans.C, Trans.C, C.numRows(), C.numColumns(), rows, alphar, alphai, a, Math.max(1, rows),
                B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA_R, BETA_I, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransAmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        Checks.checkTransAmultAdd(this, B, C);

        BlasExt blas = BlasExt.getInstance();
        blas.cgemm3m(Trans.C, Trans.N, C.numRows(), C.numColumns(), rows, alphar, alphai, a, Math.max(1, rows),
                B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA_R, BETA_I, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF conjTransBmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C) {
        Checks.checkTransBmultAdd(this, B, C);

        BlasExt blas = BlasExt.getInstance();
        blas.cgemm3m(Trans.N, Trans.C, C.numRows(), C.numColumns(), cols, alphar, alphai, a, Math.max(1, rows),
                B.getArrayUnsafe(), Math.max(1, B.numRows()), BETA_R, BETA_I, C.getArrayUnsafe(),
                Math.max(1, C.numRows()));

        return C;
    }

    // TODO ...

    /**
     * {@inheritDoc}
     */
    @Override
    public ComplexMatrixF copy() {
        return new SimpleComplexMatrixF(this);
    }
}
