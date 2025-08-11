st.caption(f"Showing {len(df_f)} of {len(df)} rows")
with st.expander("Raw fetched rows (debug)"):
    st.dataframe(df, use_container_width=True, height=280)
