<template>
    <div>
        <div v-if="!predicted" class="camera">
            <div class="header">
                <div class="header-text">Find your Place to Visit</div>
                <div class="sub-header">Barcelona</div>
            </div>
            <g-loading v-if="isLoading" class="loading"/>
            <div v-else class="camera-block">
                <button class="camera-button">Take a photo</button>
                <input type="file" accept="image/*" capture="camera" v-on:input="postImage()"
                       @keyup.enter="postImage" class="camera-input" name="image" ref="image">
            </div>
            <div v-if="errorMessage" class="error">Error: {{ errorMessage }}</div>
        </div>
        <Results v-else :predicted="predicted" v-on:reset="doReset"/>
    </div>
</template>

<script>
    import GLoading from "./GLoading.vue"
    import Results from "./Results.vue"

    const jsonfile = require('jsonfile');
    const file = 'info.json';

    export default {
        name: 'Camera',
        components: {
            GLoading,
            Results
        },
        data: function() {
            return {
                predicted: null,
                submitted: false,
                errorMessage: null,
            }
        },
        computed: {
            isLoading() {
                return this.submitted && !this.predicted;
            }
        },
        methods: {
            postImage() {
                this.doReset();
                //alert("post");
                this.submitted = true;

                this.attachImage(this.$refs.image.files[0])

                // setTimeout(() => {
                //     let class_id = 7;
                //     fetch('http://51.137.111.108:8080/info.json', {
                //         method: "GET",
                //         headers: {
                //             'Accept': 'application/json',
                //             'Content-Type': 'application/json'
                //         }
                //     }).then(data => data.json())
                //         .then(data => {
                //         console.log(data);
                //         this.predicted = data.places.find(place => place.class_id === class_id);
                //         console.log(this.predicted);
                //     })
                // }, 1000);

            },
            attachImage(file) {
                let formData = new FormData();
                formData.append('file', file);

                fetch('http://35.204.86.24', { //
                    method: 'POST',
                    headers: {
                    //     "Content-Type": "application/octet-stream"
                        //"Accept": "application/json"
                    },
                    body: formData,
                    mode: "no-cors"
                })
                    .then(response => response.json().then(function(json) {
                        this.submitted = false;
                        console.log(json);
                        let class_id = json.class;
                        alert(json.class);
                        fetch('http://51.137.111.108:8080/info.json', {
                            method: "GET",
                            headers: {
                                'Accept': 'application/json',
                                'Content-Type': 'application/json'
                            }
                        })
                            .then(data => data.json())
                            .then(data => {
                                this.predicted = data.places.find(place => place.class_id === class_id);
                            })
                    }))
                    .catch(error => {
                        this.errorMessage = error.message;
                        this.submitted = false;
                    });
                    // .then(data => {
                    //     this.predicted = data;
                    //     console.log(data);
                    // });
            },
            doReset() {
                this.predicted = null;
                this.submitted = false;
                this.errorMessage = null;
            }
        }
    }
</script>

<style scoped>
    .camera {
        background: black url('http://51.137.111.108:8080/background.jpg') no-repeat;
        background-size: 160%;
        height: 50em;
    }

    .header {
        font-family: 'Segoe UI Light', Helvetica, sans-serif;
        height: 6em;
        background-color: #FF3D00;
        opacity: 0.6;
        color: #ffffff;
    }

    .error {
        color: #ee000c;
        font-size: 18px;
    }

    .header-text {
        font-family: 'Segoe UI Light', Helvetica, sans-serif;
        text-align: left;
        padding: 0.6rem 1rem 0;
        font-size: 30px;
    }

    .sub-header {
        text-align: left;
        padding: 0 1rem 0;
        font-size: 22px;
        opacity: 0.8;
    }

    .camera-block {
        padding: 8em 3em 3em;
        position: relative;
        overflow: hidden;
        display: inline-block;
    }

    .camera-button {
        border: 0;
        background-color: #FF3D00;
        opacity: 0.6;
        color: #ffffff;
        padding: 0.4em 0.8em 0.6em;
        border-radius: 0.5em;
        font-family: 'Segoe UI Light', Helvetica, sans-serif;
        font-size: 28px;
    }

    .camera-input {
        margin: 8em 3em 3em;
        position: absolute;
        left: 0;
        top: 0;
        opacity: 0;
        height: 8em;
        width: 16em;
    }

    .loading {
        padding: 8em 3em 3em;
    }
</style>
