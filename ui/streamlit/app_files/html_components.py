import streamlit as st


def toggle_html_navbar():
    st.markdown(
        """<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
           <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css" integrity="sha512-MV7K8+y+gLIBoVD59lQIYicR65iaqukzvf/nwasF0nqhPay5w/9lJmVM2hMDcnK1OnMGCdVK+iQrJ7lzPJQd1w==" crossorigin="anonymous" referrerpolicy="no-referrer" />

        """,
        unsafe_allow_html=True,
    )
    #  <span style="color:#F68F1D;">T</span>iger<span style="color:#F68F1D;">N</span>LP


    st.markdown(
        """
        <nav class="navbar fixed-top navbar-expand-lg navbar-dark"  style="background-color: rgb(36, 41, 47);display:flex;justify-content:space-between;">
           <div>
            <a target="_self" class="navbar-brand" href="?nav=semantic_search">
            <img src="https://www.tigeranalytics.com/wp-content/uploads/logo.png" style="width:100px; height:45px;background:white;" class="navImg"/>
           </a>
           </div>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav" style="position:absolute;right:0;">
                <ul class="navbar-nav">
               </ul>
            </div>
        </nav>
    """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """,
        unsafe_allow_html=True,
    )
