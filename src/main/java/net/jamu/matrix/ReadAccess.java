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

/**
 * Reads the arrays of matrix arguments; a view is read from a copy.
 */
final class ReadAccess {

    /**
     * Returns the array to read {@code m} from, a fresh copy if {@code m} is a
     * view.
     */
    static double[] array(MatrixD m) {
        return (m instanceof MatrixDView) ? m.copy().getArrayUnsafe() : m.getArrayUnsafe();
    }

    /**
     * Returns the array to read {@code m} from, a fresh copy if {@code m} is a
     * view.
     */
    static float[] array(MatrixF m) {
        return (m instanceof MatrixFView) ? m.copy().getArrayUnsafe() : m.getArrayUnsafe();
    }

    /**
     * Returns the array to read {@code m} from, a fresh copy if {@code m} is a
     * view.
     */
    static double[] array(ComplexMatrixD m) {
        return (m instanceof ComplexMatrixDView) ? m.copy().getArrayUnsafe() : m.getArrayUnsafe();
    }

    /**
     * Returns the array to read {@code m} from, a fresh copy if {@code m} is a
     * view.
     */
    static float[] array(ComplexMatrixF m) {
        return (m instanceof ComplexMatrixFView) ? m.copy().getArrayUnsafe() : m.getArrayUnsafe();
    }

    /**
     * Returns {@code m}, or a copy if {@code m} is a view on {@code out}.
     */
    static MatrixD detach(MatrixD m, MatrixD out) {
        return (m instanceof MatrixDView && ((MatrixDView) m).shares(out.getArrayUnsafe())) ? m.copy() : m;
    }

    /**
     * Returns {@code m}, or a copy if {@code m} is a view on {@code out}.
     */
    static MatrixF detach(MatrixF m, MatrixF out) {
        return (m instanceof MatrixFView && ((MatrixFView) m).shares(out.getArrayUnsafe())) ? m.copy() : m;
    }

    /**
     * Returns {@code m}, or a copy if {@code m} is a view on {@code out}.
     */
    static ComplexMatrixD detach(ComplexMatrixD m, ComplexMatrixD out) {
        return (m instanceof ComplexMatrixDView && ((ComplexMatrixDView) m).shares(out.getArrayUnsafe())) ? m.copy()
                : m;
    }

    /**
     * Returns {@code m}, or a copy if {@code m} is a view on {@code out}.
     */
    static ComplexMatrixF detach(ComplexMatrixF m, ComplexMatrixF out) {
        return (m instanceof ComplexMatrixFView && ((ComplexMatrixFView) m).shares(out.getArrayUnsafe())) ? m.copy()
                : m;
    }

    /**
     * A {@code MatrixD} as gemm reads it: array, offset and leading dimension.
     */
    static final class OperandD {
        final double[] array;
        final int offset;
        final int ld;

        OperandD(double[] array, int offset, int ld) {
            this.array = array;
            this.offset = offset;
            this.ld = ld;
        }
    }

    /**
     * A {@code MatrixF} as gemm reads it: array, offset and leading dimension.
     */
    static final class OperandF {
        final float[] array;
        final int offset;
        final int ld;

        OperandF(float[] array, int offset, int ld) {
            this.array = array;
            this.offset = offset;
            this.ld = ld;
        }
    }

    /**
     * Returns {@code m} as a gemm operand, read from a copy if it is a view on
     * {@code out}.
     */
    static OperandD operand(MatrixD m, double[] out) {
        if (m instanceof MatrixDView) {
            return ((MatrixDView) m).operand(out);
        }
        return new OperandD(m.getArrayUnsafe(), 0, Math.max(1, m.numRows()));
    }

    /**
     * Returns {@code m} as a gemm operand, read from a copy if it is a view on
     * {@code out}.
     */
    static OperandF operand(MatrixF m, float[] out) {
        if (m instanceof MatrixFView) {
            return ((MatrixFView) m).operand(out);
        }
        return new OperandF(m.getArrayUnsafe(), 0, Math.max(1, m.numRows()));
    }

    /**
     * Returns {@code m} as a zgemm operand at offset 0, read from a copy if it
     * is a view on {@code out} or not anchored at {@code (0, 0)}.
     */
    static OperandD operand(ComplexMatrixD m, double[] out) {
        if (m instanceof ComplexMatrixDView) {
            return ((ComplexMatrixDView) m).operand(out);
        }
        return new OperandD(m.getArrayUnsafe(), 0, Math.max(1, m.numRows()));
    }

    /**
     * Returns {@code m} as a cgemm operand at offset 0, read from a copy if it
     * is a view on {@code out} or not anchored at {@code (0, 0)}.
     */
    static OperandF operand(ComplexMatrixF m, float[] out) {
        if (m instanceof ComplexMatrixFView) {
            return ((ComplexMatrixFView) m).operand(out);
        }
        return new OperandF(m.getArrayUnsafe(), 0, Math.max(1, m.numRows()));
    }

    private ReadAccess() {
        throw new AssertionError();
    }
}
