# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.models import variant_scoring_utils
import anndata
import numpy as np
import pandas as pd


def _assert_anndata_equal(result: anndata.AnnData, expected: anndata.AnnData):
  np.testing.assert_array_equal(result.X, expected.X)
  pd.testing.assert_frame_equal(result.obs, expected.obs)
  pd.testing.assert_frame_equal(result.var, expected.var)
  for k, v in expected.layers.items():
    np.testing.assert_array_equal(result.layers[k], v)


class VariantScoringUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Empty',
          scores=anndata.AnnData(),
          expected=anndata.AnnData(),
      ),
      dict(
          testcase_name='EmptyGeneScores',
          scores=anndata.AnnData(
              X=np.zeros((0, 2), dtype=np.float32),
              obs=pd.DataFrame({'gene_id': [], 'strand': []}, index=[]),
              var=pd.DataFrame(
                  {'name': ['track1', 'track1'], 'strand': ['+', '-']},
                  index=['0', '1'],
              ),
          ),
          expected=anndata.AnnData(
              X=np.zeros((0, 1), dtype=np.float32),
              obs=pd.DataFrame({'gene_id': [], 'strand': []}, index=[]),
              var=pd.DataFrame(
                  {'name': ['track1'], 'strand': ['.']}, index=['0']
              ),
          ),
      ),
      dict(
          testcase_name='NoGeneScores',
          scores=anndata.AnnData(
              X=np.zeros((0, 2), dtype=np.float32),
              obs=pd.DataFrame(index=[]),
              var=pd.DataFrame(
                  {'name': ['track1', 'track1'], 'strand': ['+', '-']},
                  index=['0', '1'],
              ),
          ),
          expected=anndata.AnnData(
              X=np.zeros((0, 2), dtype=np.float32),
              obs=pd.DataFrame(index=[]),
              var=pd.DataFrame(
                  {'name': ['track1', 'track1'], 'strand': ['+', '-']},
                  index=['0', '1'],
              ),
          ),
      ),
      dict(
          testcase_name='MergeGeneScores',
          scores=anndata.AnnData(
              X=np.array(
                  [
                      [np.nan, 1.0, 2.0],
                      [3.0, np.nan, 4.0],
                      [5.0, np.nan, 6.0],
                  ],
                  dtype=np.float32,
              ),
              obs=pd.DataFrame(
                  {
                      'gene_id': ['gene_a', 'gene_b', 'gene_c'],
                      'strand': ['-', '+', '+'],
                  },
                  index=['0', '1', '2'],
              ),
              var=pd.DataFrame(
                  {
                      'name': ['track1', 'track1', 'track2'],
                      'strand': ['+', '-', '.'],
                  },
                  index=['0', '1', '2'],
              ),
          ),
          expected=anndata.AnnData(
              X=np.arange(1, 7, dtype=np.float32).reshape((3, 2)),
              obs=pd.DataFrame(
                  {
                      'gene_id': ['gene_a', 'gene_b', 'gene_c'],
                      'strand': ['-', '+', '+'],
                  },
                  index=['0', '1', '2'],
              ),
              var=pd.DataFrame(
                  {'name': ['track1', 'track2'], 'strand': '.'},
                  index=['0', '1'],
              ),
          ),
      ),
      dict(
          testcase_name='MergeGeneScoresNotInterleaved',
          scores=anndata.AnnData(
              X=np.array(
                  [[0, 1, np.nan, np.nan, 4], [np.nan, np.nan, 2, 3, 4]],
                  dtype=np.float32,
              ),
              obs=pd.DataFrame(
                  {'gene_id': ['gene_a', 'gene_b'], 'strand': ['+', '-']},
                  index=['0', '1'],
              ),
              var=pd.DataFrame(
                  {
                      'name': ['t1', 't2', 't1', 't2', 't3'],
                      'strand': ['+', '+', '-', '-', '.'],
                  },
                  index=['0', '1', '2', '3', '4'],
              ),
          ),
          expected=anndata.AnnData(
              X=np.array([[0.0, 1.0, 4.0], [2.0, 3.0, 4.0]], dtype=np.float32),
              obs=pd.DataFrame(
                  {'gene_id': ['gene_a', 'gene_b'], 'strand': ['+', '-']},
                  index=['0', '1'],
              ),
              var=pd.DataFrame(
                  {'name': ['t1', 't2', 't3'], 'strand': '.'},
                  index=['0', '1', '2'],
              ),
          ),
      ),
      dict(
          testcase_name='MergeGeneScoresWithLayers',
          scores=anndata.AnnData(
              X=np.array(
                  [[np.nan, 1.0, 2.0], [3.0, np.nan, 5.0], [6.0, np.nan, 8.0]]
              ),
              obs=pd.DataFrame(
                  {
                      'gene_id': ['gene_a', 'gene_b', 'gene_c'],
                      'strand': ['-', '+', '+'],
                  },
                  index=['0', '1', '2'],
              ),
              var=pd.DataFrame(
                  {
                      'name': ['track1', 'track1', 'track2'],
                      'strand': ['+', '-', '.'],
                  },
                  index=['0', '1', '2'],
              ),
              layers={
                  'quantiles': np.array([
                      [np.nan, 2.0, 3.0],
                      [4.0, np.nan, 6.0],
                      [7.0, np.nan, 9.0],
                  ]),
              },
          ),
          expected=anndata.AnnData(
              X=np.array(
                  [[1.0, 2.0], [3.0, 5.0], [6.0, 8.0]], dtype=np.float32
              ),
              obs=pd.DataFrame(
                  {
                      'gene_id': ['gene_a', 'gene_b', 'gene_c'],
                      'strand': ['-', '+', '+'],
                  },
                  index=['0', '1', '2'],
              ),
              var=pd.DataFrame(
                  {'name': ['track1', 'track2'], 'strand': '.'},
                  index=['0', '1'],
              ),
              layers={
                  'quantiles': np.array(
                      [[2.0, 3.0], [4.0, 6.0], [7.0, 9.0]], dtype=np.float32
                  ),
              },
          ),
      ),
      dict(
          testcase_name='MergeGeneScoresWithUnstrandedTracks',
          scores=anndata.AnnData(
              X=np.arange(9, dtype=np.float32).reshape((3, 3)),
              obs=pd.DataFrame(
                  {
                      'gene_id': ['gene_a', 'gene_b', 'gene_c'],
                      'strand': ['-', '+', '-'],
                  },
                  index=['0', '1', '2'],
              ),
              var=pd.DataFrame(
                  {
                      'name': ['track1', 'track2', 'track3'],
                      'strand': '.',
                  },
                  index=['0', '1', '2'],
              ),
          ),
          expected=anndata.AnnData(
              X=np.arange(9, dtype=np.float32).reshape((3, 3)),
              obs=pd.DataFrame(
                  {
                      'gene_id': ['gene_a', 'gene_b', 'gene_c'],
                      'strand': ['-', '+', '-'],
                  },
                  index=['0', '1', '2'],
              ),
              var=pd.DataFrame(
                  {'name': ['track1', 'track2', 'track3'], 'strand': '.'},
                  index=['0', '1', '2'],
              ),
          ),
      ),
      dict(
          testcase_name='DropDuplicateUnstrandedTracks',
          scores=anndata.AnnData(
              X=np.array(
                  [
                      [np.nan, 1.0, 2.0],
                      [3.0, np.nan, 4.0],
                      [5.0, np.nan, 6.0],
                  ],
              ),
              obs=pd.DataFrame(
                  {
                      'gene_id': ['gene_a', 'gene_b', 'gene_c'],
                      'strand': ['-', '+', '+'],
                  },
                  index=['0', '1', '2'],
              ),
              var=pd.DataFrame(
                  {
                      'name': ['track1', 'track1', 'track1'],
                      'strand': ['+', '-', '.'],
                  },
                  index=['0', '1', '2'],
              ),
          ),
          expected=anndata.AnnData(
              X=np.array([[1.0], [3.0], [5.0]]),
              obs=pd.DataFrame(
                  {
                      'gene_id': ['gene_a', 'gene_b', 'gene_c'],
                      'strand': ['-', '+', '+'],
                  },
                  index=['0', '1', '2'],
              ),
              var=pd.DataFrame(
                  {'name': ['track1'], 'strand': '.'},
                  index=['0'],
              ),
          ),
          expected_unmerged=anndata.AnnData(
              X=np.array(
                  [
                      [np.nan, 1.0, 1.0],
                      [3.0, np.nan, 3.0],
                      [5.0, np.nan, 5.0],
                  ],
              ),
              obs=pd.DataFrame(
                  {
                      'gene_id': ['gene_a', 'gene_b', 'gene_c'],
                      'strand': ['-', '+', '+'],
                  },
                  index=['0', '1', '2'],
              ),
              var=pd.DataFrame(
                  {
                      'name': ['track1', 'track1', 'track1'],
                      'strand': ['+', '-', '.'],
                  },
                  index=['0', '1', '2'],
              ),
          ),
      ),
  )
  def test_merge_stranded_gene_tracks(
      self,
      scores: anndata.AnnData,
      expected: anndata.AnnData,
      expected_unmerged: anndata.AnnData | None = None,
  ):
    result = variant_scoring_utils.merge_stranded_gene_tracks(scores)
    _assert_anndata_equal(result, expected)

    round_trip = variant_scoring_utils.unmerge_stranded_gene_tracks(
        result, track_metadata=scores.var
    )
    expected_unmerged = expected_unmerged or scores
    _assert_anndata_equal(round_trip, expected_unmerged)

  @parameterized.named_parameters(
      dict(
          testcase_name='Empty',
          metadata=pd.DataFrame({'name': [], 'strand': '.'}),
          expected=pd.DataFrame({'name': [], 'strand': '.'}),
      ),
      dict(
          testcase_name='MergeStrandedTracks',
          metadata=pd.DataFrame({
              'name': ['track1', 'track1', 'track2'],
              'strand': ['+', '-', '.'],
          }),
          expected=pd.DataFrame({'name': ['track1', 'track2'], 'strand': '.'}),
      ),
      dict(
          testcase_name='MergeDuplicateUnstrandedTracks',
          metadata=pd.DataFrame({
              'name': ['track1', 'track1', 'track1', 'track2'],
              'strand': ['+', '-', '.', '.'],
          }),
          expected=pd.DataFrame({'name': ['track1', 'track2'], 'strand': '.'}),
      ),
  )
  def test_merge_stranded_track_metadata(
      self,
      metadata: pd.DataFrame,
      expected: pd.DataFrame,
  ):
    result = variant_scoring_utils.merge_stranded_track_metadata(metadata)
    pd.testing.assert_frame_equal(result, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='TrackMismatch',
          scores=anndata.AnnData(
              X=np.zeros((1, 3), dtype=np.float32),
              obs=pd.DataFrame(
                  {'gene_id': ['gene1'], 'strand': ['+']}, index=['0']
              ),
              var=pd.DataFrame(
                  {
                      'name': ['track1', 'track2', 'track3'],
                      'strand': ['+', '-', '.'],
                  },
                  index=['0', '1', '2'],
              ),
          ),
          expected_error=(
              'Positive and negative tracks do not match. positive_track_names:'
              r" \['track1'\], negative_track_names: \['track2'\]"
          ),
      ),
      dict(
          testcase_name='WrongOrder',
          scores=anndata.AnnData(
              X=np.zeros((1, 4), dtype=np.float32),
              obs=pd.DataFrame(
                  {'gene_id': ['gene1'], 'strand': ['+']}, index=['0']
              ),
              var=pd.DataFrame(
                  {
                      'name': ['t1', 't2', 't2', 't1'],
                      'strand': ['+', '-', '+', '-'],
                  },
                  index=['0', '1', '2', '3'],
              ),
          ),
          expected_error=(
              'Positive and negative tracks do not match. positive_track_names:'
              r" \['t1', 't2'\], negative_track_names: \['t2', 't1'\]"
          ),
      ),
      dict(
          testcase_name='MissingTracks',
          scores=anndata.AnnData(
              X=np.zeros((1, 3), dtype=np.float32),
              obs=pd.DataFrame(
                  {'gene_id': ['gene1'], 'strand': ['+']}, index=['0']
              ),
              var=pd.DataFrame(
                  {
                      'name': ['t1', 't2', 't1'],
                      'strand': ['+', '+', '-'],
                  },
                  index=['0', '1', '2'],
              ),
          ),
          expected_error=(
              'Positive and negative tracks do not match. positive_track_names:'
              r" \['t1', 't2'\], negative_track_names: \['t1'\]"
          ),
      ),
      dict(
          testcase_name='MissingTrackNames',
          scores=anndata.AnnData(
              X=np.zeros((1, 3), dtype=np.float32),
              obs=pd.DataFrame(
                  {'gene_id': ['gene1'], 'strand': ['+']}, index=['0']
              ),
              var=pd.DataFrame(
                  {
                      'strand': ['+', '+', '-'],
                  },
                  index=['0', '1', '2'],
              ),
          ),
          expected_error=(
              'Track metadata must contain "name" and "strand" columns.'
          ),
      ),
  )
  def test_merge_stranded_gene_tracks_strand_mismatch_raises_error(
      self, scores: anndata.AnnData, expected_error: str
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      variant_scoring_utils.merge_stranded_gene_tracks(scores)

  @parameterized.named_parameters(
      dict(
          testcase_name='MissingTrackNames',
          scores=anndata.AnnData(
              X=np.zeros((1, 1), dtype=np.float32),
              obs=pd.DataFrame(
                  {'gene_id': ['gene1'], 'strand': ['+']}, index=['0']
              ),
              var=pd.DataFrame({'strand': '.'}, index=['0']),
          ),
          track_metadata=pd.DataFrame(
              {'name': 'track1', 'strand': '.'}, index=['0']
          ),
          expected_error=(
              'Track metadata must contain "name" and "strand" columns.'
          ),
      ),
      dict(
          testcase_name='ScoresNotUnstranded',
          scores=anndata.AnnData(
              X=np.zeros((1, 3), dtype=np.float32),
              obs=pd.DataFrame(
                  {'gene_id': ['gene1'], 'strand': ['+']}, index=['0']
              ),
              var=pd.DataFrame(
                  {
                      'name': ['t1', 't2', 't3'],
                      'strand': ['+', '.', '.'],
                  },
                  index=['0', '1', '2'],
              ),
          ),
          track_metadata=pd.DataFrame(
              {'name': ['t1', 't2', 't3'], 'strand': ['.', '.', '.']},
              index=['0', '1', '2'],
          ),
          expected_error='Scores must have all unstranded tracks to unmerge.',
      ),
      dict(
          testcase_name='MissingStrand',
          scores=anndata.AnnData(
              X=np.zeros((1, 3), dtype=np.float32),
              obs=pd.DataFrame(
                  {'gene_id': ['gene1'], 'strand': ['+']}, index=['0']
              ),
              var=pd.DataFrame(
                  {
                      'name': ['t1', 't2', 't3'],
                      'strand': ['.', '.', '.'],
                  },
                  index=['0', '1', '2'],
              ),
          ),
          track_metadata=pd.DataFrame(
              {'name': ['t1', 't2', 't3']}, index=['0', '1', '2']
          ),
          expected_error=(
              'Track metadata must contain "name" and "strand" columns.'
          ),
      ),
      dict(
          testcase_name='MissingTracks',
          scores=anndata.AnnData(
              X=np.zeros((1, 2), dtype=np.float32),
              obs=pd.DataFrame(
                  {'gene_id': ['gene1'], 'strand': ['+']}, index=['0']
              ),
              var=pd.DataFrame(
                  {
                      'name': ['t1', 't2'],
                      'strand': ['.', '.'],
                  },
                  index=['0', '1'],
              ),
          ),
          track_metadata=pd.DataFrame(
              {'name': ['t1', 't2', 't3'], 'strand': ['.', '.', '.']},
              index=['0', '1', '2'],
          ),
          expected_error=(
              r"Scores missing tracks to unmerge! Missing tracks: \['t3'\]"
          ),
      ),
  )
  def test_unmerge_stranded_gene_tracks_mismatch_raises_error(
      self,
      scores: anndata.AnnData,
      track_metadata: pd.DataFrame,
      expected_error: str,
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      variant_scoring_utils.unmerge_stranded_gene_tracks(
          scores, track_metadata=track_metadata
      )


if __name__ == '__main__':
  absltest.main()
