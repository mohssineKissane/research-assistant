"""
Quick verification script for Agent & Tools feature
"""
import sys

def verify_imports():
    """Verify all agent and tool imports work"""
    print("Verifying imports...")
    
    try:
        from src.agent.research_agent import ResearchAgent
        print("  [OK] ResearchAgent imported")
    except Exception as e:
        print(f"  [FAIL] ResearchAgent import failed: {e}")
        return False
    
    try:
        from src.agent.agent_config import AgentConfig
        print("  [OK] AgentConfig imported")
    except Exception as e:
        print(f"  [FAIL] AgentConfig import failed: {e}")
        return False
    
    try:
        from src.tools.document_search import DocumentSearchTool
        print("  [OK] DocumentSearchTool imported")
    except Exception as e:
        print(f"  [FAIL] DocumentSearchTool import failed: {e}")
        return False
    
    try:
        from src.tools.web_search import WebSearchTool
        print("  [OK] WebSearchTool imported")
    except Exception as e:
        print(f"  [FAIL] WebSearchTool import failed: {e}")
        return False
    
    try:
        from src.tools.summarization import SummarizationTool
        print("  [OK] SummarizationTool imported")
    except Exception as e:
        print(f"  [FAIL] SummarizationTool import failed: {e}")
        return False
    
    try:
        from src.main import ResearchAssistant
        print("  [OK] ResearchAssistant imported")
    except Exception as e:
        print(f"  [FAIL] ResearchAssistant import failed: {e}")
        return False
    
    return True

def verify_integration():
    """Verify agent is integrated into ResearchAssistant"""
    print("\n Verifying integration...")
    
    try:
        from src.main import ResearchAssistant
        
        # Check if ResearchAssistant has agent methods
        assistant = ResearchAssistant()
        
        if not hasattr(assistant, 'setup_agent'):
            print("  [FAIL] setup_agent method not found")
            return False
        print("  [OK] setup_agent method exists")
        
        if not hasattr(assistant, 'ask_agent'):
            print("  [FAIL] ask_agent method not found")
            return False
        print("  [OK] ask_agent method exists")
        
        if not hasattr(assistant, 'agent'):
            print("  [FAIL] agent attribute not found")
            return False
        print("  [OK] agent attribute exists")
        
        if not hasattr(assistant, 'agent_config'):
            print("  [FAIL] agent_config attribute not found")
            return False
        print("  [OK] agent_config attribute exists")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Integration verification failed: {e}")
        return False

def verify_config():
    """Verify AgentConfig works correctly"""
    print("\n Verifying AgentConfig...")
    
    try:
        from src.agent.agent_config import AgentConfig
        
        config = AgentConfig()
        
        # Check default values
        assert config.agent_type == "zero-shot-react-description"
        print("  [OK] Default agent_type correct")
        
        assert config.verbose == True
        print("  [OK] Default verbose correct")
        
        assert config.max_iterations == 5
        print("  [OK] Default max_iterations correct")
        
        assert config.temperature == 0.7
        print("  [OK] Default temperature correct")
        
        # Check methods exist
        assert hasattr(config, 'get_agent_kwargs')
        print("  [OK] get_agent_kwargs method exists")
        
        assert hasattr(AgentConfig, 'get_research_agent_prefix')
        print("  [OK] get_research_agent_prefix method exists")
        
        assert hasattr(AgentConfig, 'get_research_agent_suffix')
        print("  [OK] get_research_agent_suffix method exists")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Config verification failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Agent & Tools Feature Verification")
    print("=" * 60)
    
    results = []
    
    # Run verifications
    results.append(("Imports", verify_imports()))
    results.append(("Integration", verify_integration()))
    results.append(("Configuration", verify_config()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n All verifications passed! Agent & Tools feature is ready.")
        print("\nNext steps:")
        print("1. Run: jupyter notebook notebooks/agent_experiments.ipynb")
        print("2. Or use in code: assistant.setup_agent() and assistant.ask_agent()")
        return 0
    else:
        print("\n[WARNING]  Some verifications failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

