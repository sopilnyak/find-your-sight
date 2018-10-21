<template>
    <div id="app">
        <Camera/>
    </div>
</template>

<script>
    import Camera from './components/Camera.vue'

    export default {
        name: 'app',
        components: {
            Camera
        }
    }
</script>

<style>
    #app {
        background-color: #000000;
        width: 100%;
        height: 600px;
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        text-align: center;
        margin: 0;
    }
</style>
