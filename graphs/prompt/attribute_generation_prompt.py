def attribute_generation_prompt(entity, semantic_units, relationships):
    base = f"""
Generate a concise summary of the given entity, capturing its essential attributes and important relevant relationships. The summary should read like a character sketch in a novel or a product description, providing an engaging yet precise overview. Ensure the output only includes the summary of the entity without any additional explanations or metadata. The length must not exceed 2000 words but can be shorter if the input material is limited. Focus on distilling the most important insights with a smooth narrative flow, highlighting the entity’s core traits and meaningful connections.
--Entity--
{entity}
--Related Semantic Units--
{semantic_units}
--Related Relationships--
{relationships}
"""
    return base