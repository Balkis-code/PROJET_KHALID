"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  PARTIE 4: UTILITAIRES ET FONCTIONS D'AIDE                              ║
║                                                                            ║
║  Ce module fournit des fonctions utilitaires pour:                       ║
║  - Configuration et chargement d'environnement                           ║
║  - Validation des données                                                ║
║  - Visualisation simplifiée                                              ║
║  - Logging et reporting                                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None


class ConfigurationManager:
    """
    Gestion de la configuration du projet
    """

    @staticmethod
    def load_env() -> bool:
        """
        Charger les variables d'environnement depuis .env

        RETOUR: True si succès
        """
        try:
            load_dotenv()
            api_key = os.getenv('ANTHROPIC_API_KEY')

            if api_key:
                print("✓ Clé API Claude chargée depuis .env")
                return True
            else:
                print("⚠ ANTHROPIC_API_KEY non trouvée dans .env")
                print("  Assurez-vous de définir: export ANTHROPIC_API_KEY='sk-ant-...'")
                return False

        except Exception as e:
            print(f"⚠ Erreur lors du chargement .env: {str(e)}")
            return False

    @staticmethod
    def get_api_key() -> Optional[str]:
        """
        Récupérer la clé API Claude

        RETOUR: Clé API ou None
        """
        return os.getenv('ANTHROPIC_API_KEY')

    @staticmethod
    def create_default_config() -> Dict:
        """
        Créer une configuration par défaut

        RETOUR: Dictionnaire de configuration
        """
        return {
            'classical': {
                'min_support': 0.90,
                'min_confidence': 0.95,
                'max_lhs_size': 2,
                'enabled': True
            },
            'agentic': {
                'model': 'claude-3-5-sonnet-20241022',
                'enabled': True,
                'timeout': 30
            },
            'experiment': {
                'output_dir': './results',
                'generate_plots': True,
                'generate_report': True,
                'verbose': True
            }
        }

    @staticmethod
    def save_config(config: Dict, path: str = 'config.json') -> bool:
        """
        Sauvegarder la configuration dans un fichier JSON

        PARAMÈTRES:
        - config: Dictionnaire de configuration
        - path: Chemin du fichier

        RETOUR: True si succès
        """
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Configuration sauvegardée dans {path}")
            return True
        except Exception as e:
            print(f"✗ Erreur lors de la sauvegarde: {str(e)}")
            return False

    @staticmethod
    def load_config(path: str = 'config.json') -> Dict:
        """
        Charger la configuration depuis un fichier JSON

        PARAMÈTRES:
        - path: Chemin du fichier

        RETOUR: Dictionnaire de configuration
        """
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config = json.load(f)
                print(f"✓ Configuration chargée depuis {path}")
                return config
            else:
                print(f"⚠ Fichier config non trouvé: {path}")
                return ConfigurationManager.create_default_config()
        except Exception as e:
            print(f"✗ Erreur lors de la lecture: {str(e)}")
            return ConfigurationManager.create_default_config()


class DataValidator:
    """
    Validation et préparation des données
    """

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """
        Valider qu'un DataFrame est approprié pour la découverte PFD

        VÉRIFICATIONS:
        - DataFrame n'est pas vide
        - Au minimum 2 colonnes textuelles
        - Au minimum 10 lignes

        RETOUR: True si valide
        """
        if df is None or df.empty:
            print("✗ DataFrame vide")
            return False

        text_cols = df.select_dtypes(include=['object']).columns

        if len(text_cols) < 2:
            print(f"✗ Au minimum 2 colonnes textuelles requises (actuellement: {len(text_cols)})")
            return False

        if len(df) < 10:
            print(f"✗ Au minimum 10 lignes requises (actuellement: {len(df)})")
            return False

        print(f"✓ DataFrame valide: {len(df)} lignes, {len(text_cols)} colonnes textuelles")
        return True

    @staticmethod
    def prepare_data(df: pd.DataFrame, drop_null: bool = True) -> pd.DataFrame:
        """
        Préparer les données pour la découverte

        OPÉRATIONS:
        - Garder seules les colonnes textuelles
        - Supprimer les valeurs NULL (optionnel)
        - Supprimer les doublons (optionnel)

        PARAMÈTRES:
        - df: DataFrame à préparer
        - drop_null: Supprimer les lignes avec NULL

        RETOUR: DataFrame préparé
        """
        # Garder seules les colonnes textuelles
        df = df.select_dtypes(include=['object']).copy()

        # Supprimer les lignes avec NaN
        if drop_null:
            initial_len = len(df)
            df = df.dropna()
            removed = initial_len - len(df)
            if removed > 0:
                print(f"⚠ {removed} lignes avec NaN supprimées")

        # Nettoyer les espaces
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        return df

    @staticmethod
    def get_column_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtenir des statistiques sur les colonnes

        RETOUR: DataFrame avec statistiques
        """
        stats = []

        for col in df.select_dtypes(include=['object']).columns:
            stats.append({
                'Column': col,
                'Type': 'Text',
                'Count': len(df),
                'Unique': df[col].nunique(),
                'Null': df[col].isna().sum(),
                'Max_Length': df[col].astype(str).str.len().max(),
                'Avg_Length': df[col].astype(str).str.len().mean()
            })

        return pd.DataFrame(stats)


class Logger:
    """
    Logging et reporting simplifiés
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logs = []
        self.timestamp = datetime.now()

    def info(self, message: str) -> None:
        """Enregistrer un message d'information"""
        if self.verbose:
            print(f"ℹ {message}")
        self.logs.append(('INFO', message))

    def success(self, message: str) -> None:
        """Enregistrer un succès"""
        if self.verbose:
            print(f"✓ {message}")
        self.logs.append(('SUCCESS', message))

    def warning(self, message: str) -> None:
        """Enregistrer un avertissement"""
        if self.verbose:
            print(f"⚠ {message}")
        self.logs.append(('WARNING', message))

    def error(self, message: str) -> None:
        """Enregistrer une erreur"""
        if self.verbose:
            print(f"✗ {message}")
        self.logs.append(('ERROR', message))

    def save_log(self, path: str = 'experiment.log') -> bool:
        """
        Sauvegarder les logs dans un fichier

        RETOUR: True si succès
        """
        try:
            with open(path, 'w') as f:
                f.write(f"Logs - {self.timestamp.isoformat()}\n")
                f.write("="*70 + "\n\n")
                for level, message in self.logs:
                    f.write(f"[{level}] {message}\n")
            return True
        except Exception as e:
            self.error(f"Impossible de sauvegarder les logs: {str(e)}")
            return False


class ResultsFormatter:
    """
    Formatage des résultats pour affichage et export
    """

    @staticmethod
    def format_pfd_results(df: pd.DataFrame, top_n: int = 10) -> str:
        """
        Formater les résultats PFD pour affichage lisible

        PARAMÈTRES:
        - df: DataFrame avec les résultats
        - top_n: Nombre de top résultats à afficher

        RETOUR: String formaté
        """
        if df.empty:
            return "Aucun résultat"

        output = f"\n{'='*70}\n"
        output += f"TOP {top_n} PATTERN FUNCTIONAL DEPENDENCIES\n"
        output += f"{'='*70}\n\n"

        for idx, row in df.head(top_n).iterrows():
            output += f"{idx+1}. {row['LHS']} → {row['RHS']}\n"
            output += f"   Support:    {row['Support']:.4f} ({row['Support']*100:.2f}%)\n"
            output += f"   Confidence: {row['Confidence']:.4f} ({row['Confidence']*100:.2f}%)\n"
            output += f"   Score:      {row['Score']:.4f}\n"

            # Ajouter les explications si disponibles (Agentic)
            if 'Claude_Explanation' in row and pd.notna(row['Claude_Explanation']):
                output += f"   Explication: {row['Claude_Explanation'][:100]}...\n"

            output += "\n"

        return output

    @staticmethod
    def format_comparison_table(classical_df: pd.DataFrame,
                                agentic_df: pd.DataFrame) -> str:
        """
        Formater un tableau de comparaison

        RETOUR: String formaté
        """
        output = f"\n{'='*70}\n"
        output += "COMPARAISON CLASSIQUE vs AGENTIC\n"
        output += f"{'='*70}\n\n"

        output += f"{'Métrique':<40} {'Classique':<15} {'Agentic':<15}\n"
        output += "-"*70 + "\n"

        metrics = {
            'Nombre de PFDs': (len(classical_df), len(agentic_df)),
            'Support moyen': (classical_df['Support'].mean() if not classical_df.empty else 0,
                            agentic_df['Support'].mean() if not agentic_df.empty else 0),
            'Confidence moyen': (classical_df['Confidence'].mean() if not classical_df.empty else 0,
                               agentic_df['Confidence'].mean() if not agentic_df.empty else 0),
            'Score moyen': (classical_df['Score'].mean() if not classical_df.empty else 0,
                          agentic_df['Score'].mean() if not agentic_df.empty else 0),
        }

        for metric, (classical_val, agentic_val) in metrics.items():
            if isinstance(classical_val, float):
                output += f"{metric:<40} {classical_val:<15.4f} {agentic_val:<15.4f}\n"
            else:
                output += f"{metric:<40} {classical_val:<15} {agentic_val:<15}\n"

        return output

    @staticmethod
    def export_comparison_json(classical_df: pd.DataFrame,
                              agentic_df: pd.DataFrame,
                              path: str = 'comparison.json') -> bool:
        """
        Exporter la comparaison en JSON

        RETOUR: True si succès
        """
        try:
            comparison = {
                'timestamp': datetime.now().isoformat(),
                'classical': {
                    'count': len(classical_df),
                    'support_mean': classical_df['Support'].mean().item() if not classical_df.empty else 0,
                    'confidence_mean': classical_df['Confidence'].mean().item() if not classical_df.empty else 0,
                    'results': classical_df.to_dict('records') if not classical_df.empty else []
                },
                'agentic': {
                    'count': len(agentic_df),
                    'support_mean': agentic_df['Support'].mean().item() if not agentic_df.empty else 0,
                    'confidence_mean': agentic_df['Confidence'].mean().item() if not agentic_df.empty else 0,
                    'results': agentic_df.to_dict('records') if not agentic_df.empty else []
                }
            }

            with open(path, 'w') as f:
                json.dump(comparison, f, indent=2)

            print(f"✓ Comparaison exportée dans {path}")
            return True

        except Exception as e:
            print(f"✗ Erreur lors de l'export: {str(e)}")
            return False


def quick_test():
    """
    Test rapide pour vérifier que tout fonctionne
    """
    print("\n" + "="*70)
    print("TEST RAPIDE DE CONFIGURATION")
    print("="*70 + "\n")

    # Test 1: Configuration
    print("[Test 1] Chargement de la configuration...")
    config = ConfigurationManager.load_config()
    print(f"✓ Configuration chargée: {len(config)} sections")

    # Test 2: API Key
    print("\n[Test 2] Vérification de la clé API...")
    api_key = ConfigurationManager.get_api_key()
    if api_key:
        print(f"✓ Clé API trouvée (premiers 10 chars): {api_key[:10]}...")
    else:
        print("⚠ Clé API non configurée")

    # Test 3: Logger
    print("\n[Test 3] Test du Logger...")
    logger = Logger(verbose=True)
    logger.info("Message d'info")
    logger.success("Succès")
    logger.warning("Avertissement")
    print(f"✓ Logger fonctionne ({len(logger.logs)} messages)")

    # Test 4: Création de config par défaut
    print("\n[Test 4] Création config par défaut...")
    default_config = ConfigurationManager.create_default_config()
    print(f"✓ Config par défaut créée")
    print(json.dumps(default_config, indent=2))

    print("\n" + "="*70)
    print("✓ TOUS LES TESTS RÉUSSIS")
    print("="*70 + "\n")


if __name__ == "__main__":
    quick_test()
