"""
Multi-algorithm Anomaly Detection Module
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore")


class AnomalyDetector:
    """
    Multi-algorithm anomaly detection for GSC/GA4 data

    Supports:
    - Local Outlier Factor (LOF)
    - Isolation Forest
    - Z-Score
    - IQR (Interquartile Range)
    - DBSCAN
    """

    def __init__(self, contamination: float = 0.05):
        """
        Initialize the anomaly detector

        Args:
            contamination: Expected proportion of anomalies (0-0.5)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()

    def _prepare_features(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        add_derived: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for anomaly detection

        Args:
            df: Input DataFrame
            metric_cols: Columns to use as features
            add_derived: Whether to add rolling statistics

        Returns:
            Feature DataFrame and scaled numpy array
        """
        df_feat = df.copy()

        if add_derived:
            for col in metric_cols:
                if col in df_feat.columns:
                    # Rolling statistics
                    df_feat[f'{col}_rolling_mean_7'] = df_feat[col].rolling(7, min_periods=1).mean()
                    df_feat[f'{col}_rolling_std_7'] = df_feat[col].rolling(7, min_periods=1).std().fillna(0)

                    # Deviation from rolling mean
                    df_feat[f'{col}_deviation'] = (
                        (df_feat[col] - df_feat[f'{col}_rolling_mean_7']) /
                        (df_feat[f'{col}_rolling_std_7'] + 1e-5)
                    )

                    # Day-over-day change
                    df_feat[f'{col}_pct_change'] = df_feat[col].pct_change().fillna(0)

        # Select numeric columns for features
        feature_cols = [c for c in df_feat.columns if df_feat[c].dtype in ['int64', 'float64'] and c != 'date']

        # Fill any remaining NaN values
        df_feat[feature_cols] = df_feat[feature_cols].fillna(0)

        # Replace infinite values
        df_feat[feature_cols] = df_feat[feature_cols].replace([np.inf, -np.inf], 0)

        # Scale features
        X_scaled = self.scaler.fit_transform(df_feat[feature_cols])

        return df_feat, X_scaled, feature_cols

    def detect_lof(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        n_neighbors: int = 20,
        add_derived: bool = True
    ) -> pd.DataFrame:
        """
        Detect anomalies using Local Outlier Factor

        Args:
            df: Input DataFrame with date and metrics
            metric_cols: Metric columns to analyze
            n_neighbors: Number of neighbors for LOF
            add_derived: Whether to add derived features

        Returns:
            DataFrame with anomaly flags and scores
        """
        df_feat, X_scaled, feature_cols = self._prepare_features(df, metric_cols, add_derived)

        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(df) - 1),
            contamination=self.contamination,
            n_jobs=-1
        )

        predictions = lof.fit_predict(X_scaled)
        scores = -lof.negative_outlier_factor_

        df_feat['is_anomaly_lof'] = predictions == -1
        df_feat['anomaly_score_lof'] = scores

        return df_feat

    def detect_isolation_forest(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        n_estimators: int = 100,
        add_derived: bool = True
    ) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest

        Args:
            df: Input DataFrame with date and metrics
            metric_cols: Metric columns to analyze
            n_estimators: Number of trees in the forest
            add_derived: Whether to add derived features

        Returns:
            DataFrame with anomaly flags and scores
        """
        df_feat, X_scaled, feature_cols = self._prepare_features(df, metric_cols, add_derived)

        iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )

        predictions = iso_forest.fit_predict(X_scaled)
        scores = -iso_forest.decision_function(X_scaled)

        df_feat['is_anomaly_iforest'] = predictions == -1
        df_feat['anomaly_score_iforest'] = scores

        return df_feat

    def detect_zscore(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect anomalies using Z-Score method

        Args:
            df: Input DataFrame with date and metrics
            metric_cols: Metric columns to analyze
            threshold: Z-score threshold for anomaly

        Returns:
            DataFrame with anomaly flags and scores
        """
        df_feat = df.copy()

        # Calculate z-scores for each metric
        z_scores = pd.DataFrame()
        for col in metric_cols:
            if col in df_feat.columns:
                z_scores[f'{col}_zscore'] = np.abs(stats.zscore(df_feat[col].fillna(0)))

        # Anomaly if any metric exceeds threshold
        df_feat['is_anomaly_zscore'] = (z_scores > threshold).any(axis=1)
        df_feat['anomaly_score_zscore'] = z_scores.max(axis=1)

        # Add z-score columns
        for col in z_scores.columns:
            df_feat[col] = z_scores[col]

        return df_feat

    def detect_iqr(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect anomalies using Interquartile Range method

        Args:
            df: Input DataFrame with date and metrics
            metric_cols: Metric columns to analyze
            multiplier: IQR multiplier for bounds

        Returns:
            DataFrame with anomaly flags
        """
        df_feat = df.copy()

        anomaly_flags = pd.DataFrame()
        iqr_scores = pd.DataFrame()

        for col in metric_cols:
            if col in df_feat.columns:
                Q1 = df_feat[col].quantile(0.25)
                Q3 = df_feat[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR

                is_outlier = (df_feat[col] < lower_bound) | (df_feat[col] > upper_bound)
                anomaly_flags[col] = is_outlier

                # Score based on distance from bounds
                score = np.where(
                    df_feat[col] < lower_bound,
                    (lower_bound - df_feat[col]) / (IQR + 1e-5),
                    np.where(
                        df_feat[col] > upper_bound,
                        (df_feat[col] - upper_bound) / (IQR + 1e-5),
                        0
                    )
                )
                iqr_scores[col] = score

        df_feat['is_anomaly_iqr'] = anomaly_flags.any(axis=1)
        df_feat['anomaly_score_iqr'] = iqr_scores.max(axis=1)

        return df_feat

    def detect_dbscan(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        eps: float = 0.5,
        min_samples: int = 5,
        add_derived: bool = True
    ) -> pd.DataFrame:
        """
        Detect anomalies using DBSCAN clustering

        Args:
            df: Input DataFrame with date and metrics
            metric_cols: Metric columns to analyze
            eps: DBSCAN epsilon parameter
            min_samples: Minimum samples for core points
            add_derived: Whether to add derived features

        Returns:
            DataFrame with anomaly flags
        """
        df_feat, X_scaled, feature_cols = self._prepare_features(df, metric_cols, add_derived)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        clusters = dbscan.fit_predict(X_scaled)

        # Points labeled as -1 are anomalies (noise)
        df_feat['is_anomaly_dbscan'] = clusters == -1
        df_feat['cluster_dbscan'] = clusters

        return df_feat

    def detect_all(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        methods: List[str] = None
    ) -> pd.DataFrame:
        """
        Run multiple anomaly detection methods

        Args:
            df: Input DataFrame with date and metrics
            metric_cols: Metric columns to analyze
            methods: List of methods to use (default: all)

        Returns:
            DataFrame with anomaly flags from all methods
        """
        methods = methods or ['lof', 'iforest', 'zscore', 'iqr']

        df_result = df.copy()

        if 'lof' in methods:
            lof_result = self.detect_lof(df, metric_cols)
            df_result['is_anomaly_lof'] = lof_result['is_anomaly_lof']
            df_result['anomaly_score_lof'] = lof_result['anomaly_score_lof']

        if 'iforest' in methods:
            iforest_result = self.detect_isolation_forest(df, metric_cols)
            df_result['is_anomaly_iforest'] = iforest_result['is_anomaly_iforest']
            df_result['anomaly_score_iforest'] = iforest_result['anomaly_score_iforest']

        if 'zscore' in methods:
            zscore_result = self.detect_zscore(df, metric_cols)
            df_result['is_anomaly_zscore'] = zscore_result['is_anomaly_zscore']
            df_result['anomaly_score_zscore'] = zscore_result['anomaly_score_zscore']

        if 'iqr' in methods:
            iqr_result = self.detect_iqr(df, metric_cols)
            df_result['is_anomaly_iqr'] = iqr_result['is_anomaly_iqr']
            df_result['anomaly_score_iqr'] = iqr_result['anomaly_score_iqr']

        if 'dbscan' in methods:
            dbscan_result = self.detect_dbscan(df, metric_cols)
            df_result['is_anomaly_dbscan'] = dbscan_result['is_anomaly_dbscan']

        # Consensus anomaly (majority vote)
        anomaly_cols = [c for c in df_result.columns if c.startswith('is_anomaly_')]
        if anomaly_cols:
            df_result['anomaly_consensus'] = df_result[anomaly_cols].sum(axis=1) >= (len(anomaly_cols) / 2)
            df_result['anomaly_vote_count'] = df_result[anomaly_cols].sum(axis=1)

        return df_result

    def classify_anomalies(
        self,
        df: pd.DataFrame,
        primary_metric: str,
        anomaly_col: str = 'anomaly_consensus'
    ) -> pd.DataFrame:
        """
        Classify anomalies as positive or negative

        Args:
            df: DataFrame with anomaly detection results
            primary_metric: Main metric to use for classification
            anomaly_col: Column with anomaly boolean flags

        Returns:
            DataFrame with anomaly_type column
        """
        df_result = df.copy()

        # Calculate deviation from rolling mean
        rolling_mean = df_result[primary_metric].rolling(7, min_periods=1).mean()
        deviation = df_result[primary_metric] - rolling_mean

        df_result['anomaly_type'] = 'Normal'

        df_result.loc[
            (df_result[anomaly_col]) & (deviation > 0),
            'anomaly_type'
        ] = 'Positive Anomaly'

        df_result.loc[
            (df_result[anomaly_col]) & (deviation <= 0),
            'anomaly_type'
        ] = 'Negative Anomaly'

        return df_result

    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for detected anomalies

        Args:
            df: DataFrame with anomaly detection results

        Returns:
            Dictionary with anomaly summary
        """
        summary = {
            'total_records': len(df),
            'methods_used': [],
            'anomalies_by_method': {},
            'consensus_anomalies': 0,
            'positive_anomalies': 0,
            'negative_anomalies': 0
        }

        # Count anomalies by method
        for col in df.columns:
            if col.startswith('is_anomaly_'):
                method = col.replace('is_anomaly_', '')
                count = df[col].sum()
                summary['methods_used'].append(method)
                summary['anomalies_by_method'][method] = int(count)

        # Consensus anomalies
        if 'anomaly_consensus' in df.columns:
            summary['consensus_anomalies'] = int(df['anomaly_consensus'].sum())

        # Classified anomalies
        if 'anomaly_type' in df.columns:
            summary['positive_anomalies'] = int((df['anomaly_type'] == 'Positive Anomaly').sum())
            summary['negative_anomalies'] = int((df['anomaly_type'] == 'Negative Anomaly').sum())

        return summary
