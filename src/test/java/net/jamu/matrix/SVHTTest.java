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

import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.junit.Test;

/**
 * Tests for SVHT, on hand built spectra rather than on decompositions, so that
 * nothing sits between the input and the answer.
 */
public final class SVHTTest {

    /** five signal values over a flat noise bulk, in 100 values */
    private static double[] referenceSpectrum() {
        double[] s = new double[100];
        for (int i = 0; i < 5; ++i) {
            s[i] = 100.0 / (i + 1);
        }
        for (int i = 5; i < s.length; ++i) {
            s[i] = 0.05;
        }
        return s;
    }

    private static double[] scaled(double[] s, double factor) {
        double[] out = new double[s.length];
        for (int i = 0; i < s.length; ++i) {
            out[i] = s[i] * factor;
        }
        return out;
    }

    private static float[] toFloat(double[] s) {
        float[] out = new float[s.length];
        for (int i = 0; i < s.length; ++i) {
            out[i] = (float) s[i];
        }
        return out;
    }

    @Test
    public void testTheAnswerOnTheReferenceSpectrum() {
        // the answer at a scale of one, which the scale invariance tests below
        // then demand everywhere else as well
        assertEquals(5, SVHT.threshold(100, 100, referenceSpectrum()));
        assertEquals(5, SVHT.threshold(100, 100, toFloat(referenceSpectrum())));
    }

    @Test
    public void testScaleInvarianceDouble() {
        // a singular value carries the scale of the matrix it came from, so the
        // answer must not depend on that scale. It used to: measured against
        // the absolute tolerance this spectrum answered 5 down to 1e-13, then
        // 1, and 0 from 1e-20 on
        for (double factor : new double[] { 1.0e+300, 1.0e+100, 1.0e+10, 1.0, 1.0e-10, 1.0e-13,
                1.0e-14, 1.0e-16, 1.0e-20, 1.0e-100, 1.0e-200, 1.0e-300 }) {
            assertEquals("scale " + factor, 5,
                    SVHT.threshold(100, 100, scaled(referenceSpectrum(), factor)));
        }
    }

    @Test
    public void testScaleInvarianceFloat() {
        // the single precision tolerance is 5 * 5.96e-8, so the collapse began
        // far earlier there: measured, at 1e-6
        for (double factor : new double[] { 1.0e+30, 1.0e+20, 1.0, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8,
                1.0e-10, 1.0e-20, 1.0e-30 }) {
            assertEquals("scale " + factor, 5,
                    SVHT.threshold(100, 100, toFloat(scaled(referenceSpectrum(), factor))));
        }
    }

    @Test
    public void testAZeroSpectrumHasNoSignal() {
        assertEquals(0, SVHT.threshold(10, 10, new double[10]));
        assertEquals(0, SVHT.threshold(10, 10, new float[10]));
    }

    @Test
    public void testANaNIsRejected() {
        // a NaN used to slip past the guard, because NaN <= eps is false, and
        // the answer came out as 1
        double[] s = referenceSpectrum();
        s[0] = Double.NaN;
        assertEquals(0, SVHT.threshold(100, 100, s));
        float[] f = toFloat(referenceSpectrum());
        f[0] = Float.NaN;
        assertEquals(0, SVHT.threshold(100, 100, f));
    }

    @Test
    public void testASpectrumWithNoBulkToMeasureAgainst() {
        // the median form needs a noise bulk to take its statistic from. Where
        // there is none the cutoff lands above every value and nothing
        // survives, which is what these three record - not a defect but the
        // nature of the criterion
        assertEquals(0, SVHT.threshold(1, 1, new double[] { 3.0 }));
        double[] flat = new double[50];
        Arrays.fill(flat, 2.0);
        assertEquals(0, SVHT.threshold(50, 50, flat));
        assertEquals(0, SVHT.threshold(10, 10, new double[] { 5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }));
    }

    @Test
    public void testTheMedianIsTakenOverTheNonNegligiblePartOnly() {
        // trailing values below the relative tolerance are dropped before the
        // median is taken, and that decision has to scale with the spectrum too
        double[] s = { 10.0, 8.0, 6.0, 4.0, 2.0, 1.0e-18, 1.0e-18, 1.0e-18 };
        assertEquals(6.0, SVHT.median(s), 0.0);
        assertEquals(6.0, SVHT.median(scaled(s, 1.0e-200)) * 1.0e+200, 1.0e-12);
    }

    @Test
    public void testTheSumSkipsTheNegligibleTailAtEveryScale() {
        double[] s = { 10.0, 8.0, 6.0, 1.0e-18, 1.0e-18 };
        assertEquals(24.0, SVHT.sum(s), 1.0e-12);
        // this used to come out as zero once the whole spectrum sat below the
        // absolute tolerance, which forced the caller to answer 1
        assertEquals(24.0, SVHT.sum(scaled(s, 1.0e-200)) * 1.0e+200, 1.0e-12);
    }
}
