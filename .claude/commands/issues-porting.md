# TensorZero Issue and Contributing Guidelines

## Contributing Process

### GitHub Issues and Discussions
- **Issues are for maintainers only** - Only for topics already discussed and approved in GitHub Discussions
- New issues automatically get the `needs-triage` label
- Community members should use GitHub Discussions for:
  - **Feature requests**: https://github.com/tensorzero/tensorzero/discussions/new?category=feature-requests
  - **Bug reports**: https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports  
  - **Help/Questions**: https://github.com/tensorzero/tensorzero/discussions/new?category=help

### Code Contributions
- **Small changes**: Open a PR directly (few lines of code)
- **Larger changes**: Communicate first via GitHub Discussions, Slack, or Discord to avoid duplicate work
- Look for `good-first-issue` label for beginner-friendly tasks

### Pull Request Template
- Must read CONTRIBUTING.md before submitting
- Must run `pre-commit` and relevant tests (including E2E tests for gateway changes)
- By submitting, you agree to Apache 2.0 license

## Issue Format Conventions

Based on existing issues in the repository:

### Issue Title
- Clear, concise description of the feature/bug
- Often starts with action verb: "Add support for...", "Standardize...", "Performance Optimization: ..."

### Issue Labels
Common labels used:
- `enhancement` - New feature or request
- `bug` - Something isn't working
- `needs-triage` - Default for new issues
- `good-first-issue` - Good for beginners

### Issue Body Structure

For **Feature Requests** (like OpenAI API support issues):

1. **Summary** - Brief overview of the feature
2. **Background** - Context and explanation of the feature
3. **API Specifications** - Detailed technical specs including:
   - Endpoints
   - Request/Response formats
   - Parameters (required and optional)
   - Example requests
4. **Technical Implementation** - How to implement:
   - Code changes needed
   - New traits/modules
   - Configuration updates
5. **Acceptance Criteria** - Checklist of requirements
6. **References** - Links to documentation
7. **Additional Notes** - Extra considerations

For **Bug Reports** (like issue #17):

1. **Problem** - Clear description of the issue
2. **Current Behavior** - What happens now
3. **Expected Behavior** - What should happen
4. **Technical Context** - Implementation details
5. **Recommended Solution** - Proposed fix with rationale
6. **Implementation Plan** - Step-by-step approach
7. **Migration Strategy** - If breaking changes involved
8. **Implementation Considerations** - Things to watch out for

### Code Formatting in Issues
- Use triple backticks with language specification for code blocks
- Use inline backticks for file paths, function names, parameters
- Include curl examples for API endpoints
- Show JSON request/response examples with proper formatting

### Best Practices
- Be thorough and detailed in technical specifications
- Include all parameters with descriptions and defaults
- Provide real curl examples that can be tested
- Break down complex features into clear sections
- Use checklists for acceptance criteria
- Reference official documentation
- Consider edge cases and limitations